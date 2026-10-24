"""Synthetic LoRA training load with Qwen3.6-27B's shapes, for the overlap prototype (PLAN.md §5b).

Each micro-batch does what Miles does per packed 48k-token micro-batch: a no-grad forward for the old
log-probs, then forward + backward with per-layer recompute, then an Adam step on the LoRA weights.
Compute and HBM traffic match the real model: `--physical` distinct layers (same linear/full-attention
pattern) are looped to make 64, and their weights are far larger than L2, so every pass re-reads them.
Memory matches too: a ballast tensor stands in for the rest of the 54 GB frozen base.
Not modelled: the gated delta-rule kernel (~2% of FLOPs; element-wise ops of the same sizes stand in).

  python -m proto.train_proxy --config /root/models/Qwen3.6-27B/config.json --minutes 4 --out train.jsonl
  python -m proto.train_proxy --config ... --pcie-only --out pcie.json
Output: JSONL, one "start" row, then one row per micro-batch (time, seconds, TFLOPS, peak memory);
an "oom" row and exit code 3 if it runs out of memory.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from proto.shapes import LORA_TARGETS, Shapes


class Linear(torch.nn.Module):
    """Frozen base weight, plus a trainable LoRA pair (alpha = rank) when rank > 0."""

    def __init__(self, i: int, o: int, rank: int, dev, dt):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(o, i, device=dev, dtype=dt) * i ** -0.5, requires_grad=False)
        self.a = torch.nn.Parameter(torch.randn(rank, i, device=dev, dtype=dt) * i ** -0.5) if rank else None
        self.b = torch.nn.Parameter(torch.randn(o, rank, device=dev, dtype=dt) * 1e-3) if rank else None

    def forward(self, x):
        y = F.linear(x, self.w)
        return y if self.a is None else y + F.linear(F.linear(x, self.a), self.b)


class Layer(torch.nn.Module):
    """One decoder layer: full attention (gated GQA) or linear attention, then the SwiGLU MLP."""

    def __init__(self, s: Shapes, full: bool, rank: int, dev, dt):
        super().__init__()
        self.s, self.full = s, full
        self.lin = torch.nn.ModuleDict({n: Linear(i, o, rank if n in LORA_TARGETS else 0, dev, dt)
                                        for n, (i, o) in s.linears(full).items()})

    def forward(self, x):
        h = F.rms_norm(x, (x.shape[-1],))
        x = x + (self.attention(h) if self.full else self.linear_attention(h))
        h = F.rms_norm(x, (x.shape[-1],))
        return x + self.lin["down_proj"](F.silu(self.lin["gate_proj"](h)) * self.lin["up_proj"](h))

    def attention(self, h):
        s, n = self.s, h.shape[0]
        q = self.lin["q_proj"](h)
        q, g = q.chunk(2, -1) if s.attn_gate else (q, None)
        heads = lambda t, k: t.view(n, k, s.head_dim).transpose(0, 1)[None]          # (1, heads, n, dim)
        k = heads(self.lin["k_proj"](h), s.kv_heads).repeat_interleave(s.heads // s.kv_heads, 1)
        v = heads(self.lin["v_proj"](h), s.kv_heads).repeat_interleave(s.heads // s.kv_heads, 1)
        o = F.scaled_dot_product_attention(heads(q, s.heads), k, v, is_causal=True)[0].transpose(0, 1)
        o = o.reshape(n, -1)
        return self.lin["o_proj"](o if g is None else o * torch.sigmoid(g))

    def linear_attention(self, h):
        """The real projections; element-wise ops on the same tensors stand in for the delta-rule kernel."""
        nk = self.s.lin_k_heads * self.s.lin_k_dim
        q, k, v = self.lin["in_proj_qkv"](h).split([nk, nk, self.s.lin_v_heads * self.s.lin_v_dim], -1)
        gate = torch.sigmoid(self.lin["in_proj_ab"](h)).mean(-1, keepdim=True)
        mix = torch.tanh((q * k).sum(-1, keepdim=True) * nk ** -0.5)
        return self.lin["out_proj"](F.silu(self.lin["in_proj_z"](h)) * v * gate * mix)


class Proxy(torch.nn.Module):
    def __init__(self, s: Shapes, physical: int, rank: int, dev, dt):
        super().__init__()
        assert s.layers % physical == 0 and physical % s.full_every == 0, "physical must keep the layer pattern"
        self.s, self.reps = s, s.layers // physical
        self.layers = torch.nn.ModuleList(Layer(s, s.is_full(k), rank, dev, dt) for k in range(physical))
        self.head = Linear(s.hidden, s.vocab, 0, dev, dt)

    def hidden(self, x, recompute: bool):
        for _ in range(self.reps):
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False) if recompute else layer(x)
        return F.rms_norm(x, (x.shape[-1],))

    def logprobs(self, h, targets, chunk: int, full_logits: bool):
        """Log-prob of each target token. Chunked: only one chunk's logits exist at a time (recomputed in
        backward). Full: all logits are kept in bf16, as Miles' output layer does today (PLAN §1)."""
        def lp(x, t, w):
            logits = x if w is None else F.linear(x, w)
            return logits.gather(-1, t[:, None])[:, 0].float() - torch.logsumexp(logits.float(), -1)
        src, w = (F.linear(h, self.head.w), None) if full_logits else (h, self.head.w)
        return torch.cat([checkpoint(lp, src[i:i + chunk], targets[i:i + chunk], w, use_reentrant=False)
                          for i in range(0, h.shape[0], chunk)])

    def base_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for n, p in self.named_parameters() if n.endswith(".w"))


def micro_batch(model: Proxy, opt, seq: int, args, dev, dt) -> None:
    """Old log-probs (no grad), then the policy-gradient forward/backward and one Adam step."""
    s = model.s
    x = torch.randn(seq, s.hidden, device=dev, dtype=dt)             # stands in for the embedded tokens
    targets = torch.randint(0, s.vocab, (seq,), device=dev)
    with torch.no_grad():
        old = model.logprobs(model.hidden(x, recompute=False), targets, args.chunk, False)
    adv = torch.randn(seq, device=dev)
    offload = torch.autograd.graph.save_on_cpu(pin_memory=dev.type == "cuda") if args.offload \
        else contextlib.nullcontext()                                # checkpointed layer inputs -> host RAM
    with offload:
        lp = model.logprobs(model.hidden(x, recompute=True), targets, args.chunk, args.logits == "full")
    (-(adv * torch.exp(lp - old)).mean()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def pcie(dev, gb: float = 4.0, reps: int = 3) -> dict:
    """Pinned host <-> GPU copy speed in GB/s (best of `reps`)."""
    host = torch.empty(int(gb * 2 ** 30), dtype=torch.uint8).pin_memory()
    gpu = torch.empty_like(host, device=dev)
    out = {}
    for name, dst, src in [("h2d_gbps", gpu, host), ("d2h_gbps", host, gpu)]:
        best = float("inf")
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        out[name] = round(gb * 2 ** 30 / best / 1e9, 1)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="model config.json (shapes)")
    p.add_argument("--out", required=True, help="JSONL output")
    p.add_argument("--seq", type=int, default=49152, help="tokens per micro-batch (48k = MAX_LEN)")
    p.add_argument("--minutes", type=float, default=4.0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--physical", type=int, default=8, help="distinct layers in memory, looped to make 64")
    p.add_argument("--chunk", type=int, default=4096, help="tokens per log-prob chunk")
    p.add_argument("--logits", choices=["chunked", "full"], default="chunked")
    p.add_argument("--offload", action="store_true", help="checkpointed layer inputs to pinned host RAM")
    p.add_argument("--ballast", default="auto", help="GB standing in for the rest of the base; auto = to 54 GB")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many micro-batches (0 = no cap)")
    p.add_argument("--pcie-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args(argv)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    s = Shapes.from_config(args.config)
    out = open(args.out, "a", buffering=1)
    emit = lambda row: out.write(json.dumps({"t": time.time(), **row}) + "\n")
    if args.pcie_only:
        emit({"type": "pcie", **pcie(dev)})
        return 0
    torch.manual_seed(0)
    model = Proxy(s, args.physical, args.rank, dev, dt)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=1e-5, betas=(0.9, 0.98))
    ballast_gb = max(0.0, s.weight_bytes() - model.base_bytes()) / 2 ** 30 if args.ballast == "auto" \
        else float(args.ballast)
    ballast = torch.empty(int(ballast_gb * 2 ** 30), dtype=torch.uint8, device=dev)   # noqa: F841 (held)
    flops = s.train_flops(args.seq)
    emit({"type": "start", "seq": args.seq, "offload": args.offload, "logits": args.logits,
          "base_gb": round(model.base_bytes() / 2 ** 30, 1), "ballast_gb": round(ballast_gb, 1),
          "lora_params": s.lora_params(args.rank), "flops_per_mb": flops,
          "mps_thread_pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")})
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    peak = (lambda: torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else (lambda: 0.0)
    end, step = time.time() + 60 * args.minutes, 0
    while time.time() < end and (not args.max_steps or step < args.max_steps):
        t0 = time.perf_counter()
        try:
            micro_batch(model, opt, args.seq, args, dev, dt)
            sync()
        except torch.OutOfMemoryError as e:
            emit({"type": "oom", "step": step, "error": str(e)[:300]})
            return 3
        dt_s = time.perf_counter() - t0
        emit({"type": "step", "step": step, "seconds": round(dt_s, 3), "tokens": args.seq,
              "tflops": round(flops / dt_s / 1e12, 1), "peak_gb": round(peak(), 1)})
        step += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
