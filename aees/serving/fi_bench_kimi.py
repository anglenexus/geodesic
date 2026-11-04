#!/usr/bin/env python3
"""
fi_bench_kimi.py - Phase 4 for Kimi K3 on Blackwell (B300): kernel microbenchmarks plus one HTML report.

    python fi_bench_kimi.py mla      # 4a: MLA decode attention, TP8 vs DCP8 layouts, BF16 vs FP8 cache (~5 min)
    python fi_bench_kimi.py fp4      # 4b: expert and attention GEMMs in BF16, FP8, MXFP8, MXFP4, NVFP4 (~5 min)
    python fi_bench_kimi.py report   # fi_report.html from whatever has run
    add --quick for a 1-minute pipeline check (dev runs on one GPU)

Shapes come from Kimi K3's config (moonshotai/Kimi-K3): 24 MLA layers with 96 heads, latent 512 + RoPE 64;
896 routed experts per MoE layer (16 per token) working in a 3584-wide latent space with 3072-wide FFNs,
stored in MXFP4 (4-bit E2M1 values, one E8M0 scale per 32); attention, shared experts and the dense layer in BF16.
Every case also records its first call (kernel load or JIT compile, 4c) next to the steady-state time.
Results land in FI_OUT (default results/fi_kimi-k3). Weight and cache values are random: only shapes and
formats matter for timing.
"""
import argparse
import html
import json
import os
import pathlib
import sys
import time

from fi_bench import bench, free_gpu, log  # same CUDA-event timing as the Qwen Phase 4

OUT = pathlib.Path(os.environ.get("FI_OUT") or pathlib.Path(os.environ.get("RESULTS_DIR", "results")) / "fi_kimi-k3")
MLA = dict(layers=24, heads=96, kv_lora_rank=512, rope=64, nope=128)
TP = 8
PEAK_GBS = {"B300": 8000, "B200": 8000, "H200": 4800, "H100": 3350}
GEMMS = [  # (name, N, K, per-token count, note)
    ("expert_gate_up", 2 * 3072, 3584, "x16 experts x 92 layers", "routed expert, MXFP4 in the checkpoint"),
    ("expert_down", 3584, 3072, "x16 experts x 92 layers", "routed expert, MXFP4 in the checkpoint"),
    ("attn_q_b_tp8", 96 * (128 + 64) // TP, 1536, "x24 MLA layers, 1/8 of the heads per GPU", "BF16 in the checkpoint"),
    ("attn_o_tp8", 7168, 96 * 128 // TP, "x24 MLA layers, 1/8 of the heads per GPU", "BF16 in the checkpoint"),
]
PRECISIONS = {"bf16": "BF16 (W16A16, cuBLAS)", "fp8": "FP8 E4M3 W8A8, per-tensor scale (cuBLASLt)",
              "mxfp8": "MXFP8 W8A8, E8M0 scale per 32", "mxfp4": "MXFP4 W4A4, E8M0 scale per 32",
              "nvfp4": "NVFP4 W4A4, E4M3 scale per 16 + FP32 per tensor"}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=str))
    log(f"saved {OUT / name}.json")


def env_info():
    import flashinfer
    import torch
    name = torch.cuda.get_device_name()
    return dict(gpu=name, peak_gbs=next((v for k, v in PEAK_GBS.items() if k in name), None),
                sm=".".join(map(str, torch.cuda.get_device_capability())),
                sms=torch.cuda.get_device_properties(0).multi_processor_count, torch=torch.__version__,
                cuda=torch.version.cuda, flashinfer=flashinfer.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                model="kimi-k3")


def timed(fn, iters=20, per_graph=10):
    """(ms per call, mode). Calls are captured in a CUDA graph, as the server runs them, so the time is the
    GPU's and not Python's or a library's per-call setup; eager timing is the fallback if capture fails."""
    import torch
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(per_graph):
                fn()
        return bench(g.replay, iters=iters, reps=5) / per_graph, "cuda graph"
    except Exception as e:
        log(f"  (graph capture failed, timing eagerly: {type(e).__name__}: {str(e)[:120]})")
        torch.cuda.synchronize()
        return bench(fn, iters=iters, reps=5), "eager"


def first_call_ms(fn):
    import torch
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


# ----------------------------------------------------------------------------- 4a MLA decode
def mla_case(B, ctx, heads, kv_dtype, page, layout, ws):
    """One decode step of one MLA layer: B requests, `ctx` cached tokens each, `heads` query heads."""
    import torch
    import flashinfer
    D = MLA["kv_lora_rank"] + MLA["rope"]
    pages_per = -(-ctx // page)
    total = B * pages_per
    if layout == "contiguous":
        ids = torch.arange(total, dtype=torch.int32)
        pool = total
    else:  # scattered over a pool 25% larger, like a churned allocator
        pool = int(total * 1.25) + 1
        ids = torch.randperm(pool, generator=torch.Generator().manual_seed(0))[:total].to(torch.int32)
    block_tables = ids.view(B, pages_per).cuda()
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
    cache = (torch.randn(pool, page, D, dtype=torch.bfloat16, device="cuda") * 0.1).to(kv_dtype)
    q = (torch.randn(B, 1, heads, D, dtype=torch.bfloat16, device="cuda") * 0.1).to(kv_dtype)
    scale = 1.0 / (MLA["nope"] + MLA["rope"]) ** 0.5

    def run():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=cache, workspace_buffer=ws, qk_nope_head_dim=MLA["nope"],
            kv_lora_rank=MLA["kv_lora_rank"], qk_rope_head_dim=MLA["rope"], block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=ctx, bmm1_scale=scale, bmm2_scale=1.0)
    first = first_call_ms(run)
    ms, mode = timed(run)
    kv_bytes = B * ctx * D * cache.element_size()
    return dict(ms=ms, timing=mode, first_ms=first, gbs=kv_bytes / ms / 1e6, kv_mb=kv_bytes / 1e6,
                per_token_attn_ms=ms * MLA["layers"])


def cmd_mla(a):
    import torch
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    shapes = [(1, 8192), (16, 8192), (64, 4096), (16, 32768), (4, 131072)]
    if a.quick:
        shapes = [(16, 8192)]
    rows = []
    for B, ctx in shapes:
        # TP8: each GPU has 1/8 of the heads and a full copy of every request's cache.
        # DCP8: each GPU has all 96 heads and 1/8 of each request's cache (partial results merged after).
        for par, heads, c in [("tp8", MLA["heads"] // TP, ctx), ("dcp8", MLA["heads"], ctx // TP)]:
            for kv_name, kv in [("bf16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn)]:
                for layout in (["contiguous", "scattered"] if kv_name == "bf16" else ["scattered"]):
                    rec = dict(batch=B, context=ctx, parallel=par, heads=heads, tokens_per_gpu=c, kv_dtype=kv_name,
                               page_size=64, layout=layout)
                    try:
                        rec.update(mla_case(B, c, heads, kv, 64, layout, ws))
                        log(f"B={B:<3} ctx={ctx:<6} {par:4s} heads={heads:<3} {kv_name:8s} {layout:10s} "
                            f"{rec['ms'] * 1000:8.1f} us  {rec['gbs']:6.0f} GB/s  first {rec['first_ms']:.0f} ms")
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"[:400]
                        log(f"B={B} ctx={ctx} {par} heads={heads} {kv_name} {layout}: {rec['error']}")
                    rows.append(rec)
                    free_gpu()
        save("mla", dict(env=env_info(), shape=MLA, cases=rows,
                         note="One MLA layer's decode attention per GPU (trtllm-gen kernel); x24 for a token."))


# ----------------------------------------------------------------------------- 4b FP4 GEMMs
def gemm_kernels(N, K):
    """{precision: (make(x) -> callable, kernel name, weight bytes)} for one (N x K) weight matrix."""
    import torch
    import torch.nn.functional as F
    import flashinfer
    out = {}
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    out["bf16"] = (lambda x: (lambda: F.linear(x, w)), "torch F.linear (cuBLAS)", 2 * N * K)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    w8 = w.to(torch.float8_e4m3fn)

    def make_fp8(x):
        xp = F.pad(x, (0, 0, 0, (-x.shape[0]) % 16)).to(torch.float8_e4m3fn)
        return lambda: torch._scaled_mm(xp, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    out["fp8"] = (make_fp8, "torch._scaled_mm (cuBLASLt FP8)", N * K)
    try:
        wq8, wsf8 = flashinfer.mxfp8_quantize(w, True)

        def make_mxfp8(x):
            xq, xsf = flashinfer.mxfp8_quantize(x, True)
            return lambda: flashinfer.mm_mxfp8(xq, wq8.t(), xsf, wsf8, out_dtype=torch.bfloat16)
        make_mxfp8(torch.randn(16, K, dtype=torch.bfloat16, device="cuda"))()
        out["mxfp8"] = (make_mxfp8, "flashinfer.mm_mxfp8", N * K + N * K // 32)
    except Exception as e:
        log(f"  MXFP8 unavailable for {N}x{K}: {type(e).__name__}: {str(e)[:200]}")
    for prec, bs, nv in [("mxfp4", 32, False), ("nvfp4", 16, True)]:
        try:
            if nv:
                gw = (448 * 6) / w.float().abs().max()
                wq, wsf = flashinfer.fp4_quantize(w, gw, sf_vec_size=16, sf_use_ue8m0=False)
            else:
                gw = None
                wq, wsf = flashinfer.fp4_quantize(w, None, sf_vec_size=32, sf_use_ue8m0=True)

            def make(x, wq=wq, wsf=wsf, gw=gw, bs=bs, nv=nv):
                if nv:
                    gx = (448 * 6) / x.float().abs().max()
                    xq, xsf = flashinfer.fp4_quantize(x, gx, sf_vec_size=16, sf_use_ue8m0=False)
                    alpha = 1.0 / (gx * gw)
                else:
                    xq, xsf = flashinfer.fp4_quantize(x, None, sf_vec_size=32, sf_use_ue8m0=True)
                    alpha = None
                return lambda: flashinfer.mm_fp4(xq, wq.t(), xsf, wsf.t(), alpha, torch.bfloat16,
                                                 block_size=bs, use_nvfp4=nv)
            make(torch.randn(16, K, dtype=torch.bfloat16, device="cuda"))()
            out[prec] = (make, f"flashinfer.mm_fp4 (block {bs})", N * K // 2 + N * K // bs)
        except Exception as e:
            log(f"  {prec} unavailable for {N}x{K}: {type(e).__name__}: {str(e)[:200]}")
    return out


def cmd_fp4(a):
    import torch
    tokens = [1, 16, 64, 256, 1024, 4096] if not a.quick else [1, 256]
    gemms = GEMMS if not a.quick else GEMMS[:1]
    rows, kernels = [], {}
    for name, N, K, count, note in gemms:
        for prec, (make, kname, wbytes) in gemm_kernels(N, K).items():
            kernels[prec] = kname
            for M in tokens:
                rec = dict(matrix=name, N=N, K=K, count=count, note=note, precision=prec, tokens=M,
                           weight_mb=wbytes / 1e6)
                try:
                    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                    fn = make(x)
                    rec["first_ms"] = first_call_ms(fn)
                    ms, mode = timed(fn, iters=10 if M >= 1024 else 20)
                    rec.update(ms=ms, timing=mode, tflops=2 * M * N * K / ms / 1e9, weight_gbs=wbytes / ms / 1e6)
                    log(f"{name:15s} {N:>5}x{K:<5} {prec:6s} M={M:<5} {ms * 1000:8.1f} us  "
                        f"{rec['tflops']:7.1f} TFLOPS  {rec['weight_gbs']:6.0f} GB/s weights")
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:400]
                    log(f"{name} {prec} M={M}: {rec['error']}")
                rows.append(rec)
            free_gpu()
        save("fp4", dict(env=env_info(), kernels=kernels, precisions=PRECISIONS, cases=rows,
                         note="One weight matrix, activations quantized outside the timed call. Kimi K3 serves its "
                              "routed experts as MXFP4 weights with MXFP8 activations in a fused MoE kernel; these "
                              "single-matrix GEMMs isolate what each number format costs."))


# ----------------------------------------------------------------------------- report
CSS = """body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:24px}h2{font-size:18px;margin-top:32px}
section{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:16px;overflow-x:auto}
.why{color:#59636e;margin:0 0 12px}.sub{color:#59636e}table{border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid #eaeef2;padding:4px 10px;text-align:right}th{background:#f6f8fa}td:first-child,th:first-child{text-align:left}"""
COLORS = {"bf16": "#1f6feb", "fp8": "#e9a23b", "mxfp8": "#bf8700", "mxfp4": "#2a9d8f", "nvfp4": "#8250df",
          "tp8": "#d1495b", "dcp8": "#1f6feb"}


def cmd_report(a):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    parts, first, env = [], [True], None

    def add(title, why, fig=None, raw=""):
        div = ""
        if fig is not None:
            fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50),
                              legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
            div = fig.to_html(full_html=False, include_plotlyjs=first[0], config={"displaylogo": False})
            first[0] = False
        parts.append(f"<h2>{html.escape(title)}</h2><section><p class='why'>{html.escape(why)}</p>{div}{raw}</section>")

    def load(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    m = load("mla")
    if m:
        env = m["env"]
        df = pd.DataFrame(m["cases"])
        df = df[df["error"].isna()] if "error" in df else df
        if len(df):
            df["shape"] = df.batch.astype(str) + " x " + (df.context // 1024).astype(str) + "K"
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Time per layer, BF16 cache (us)", "Cache read speed (GB/s)"])
            d = df[(df.kv_dtype == "bf16") & (df.layout == "scattered")]
            for par in ["tp8", "dcp8"]:
                s = d[d.parallel == par]
                fig.add_trace(go.Bar(x=s["shape"], y=s.ms * 1000, name=par.upper(), marker_color=COLORS[par]), 1, 1)
                fig.add_trace(go.Bar(x=s["shape"], y=s.gbs, name=par.upper(), marker_color=COLORS[par], showlegend=False), 1, 2)
            add("4a · MLA decode: TP8 vs DCP8",
                "One MLA layer's decode attention on one GPU. TP8 gives each GPU 12 of the 96 heads and a full copy of "
                "every request's latent cache; DCP8 gives it all 96 heads over 1/8 of the cache. Same arithmetic, "
                "8x fewer cache bytes per GPU with DCP. Batch x context on the x axis.", fig,
                raw=df[["shape", "parallel", "heads", "kv_dtype", "layout", "ms", "gbs", "first_ms"]]
                .assign(us=lambda x: (x.ms * 1000).round(1), gbs=lambda x: x.gbs.round(0), first_ms=lambda x: x.first_ms.round(0))
                .drop(columns="ms").to_html(index=False))
    f = load("fp4")
    if f:
        env = env or f["env"]
        df = pd.DataFrame(f["cases"])
        df = df[df["error"].isna()] if "error" in df else df
        if len(df):
            mats = list(dict.fromkeys(df.matrix))
            fig = make_subplots(rows=1, cols=len(mats), subplot_titles=mats)
            for j, mat in enumerate(mats, 1):
                base = df[(df.matrix == mat) & (df.precision == "bf16")].set_index("tokens").ms
                for prec in PRECISIONS:
                    s = df[(df.matrix == mat) & (df.precision == prec)].set_index("tokens").ms
                    if len(s):
                        fig.add_trace(go.Scatter(x=s.index, y=(base / s).reindex(s.index), mode="lines+markers",
                                                 name=prec, legendgroup=prec, showlegend=j == 1,
                                                 line=dict(color=COLORS[prec])), 1, j)
                fig.update_xaxes(type="log", title_text="tokens", row=1, col=j)
            fig.update_yaxes(title_text="speed vs BF16 (x)", row=1, col=1)
            add("4b · Number formats: speed vs BF16 by batch size",
                "Each line is one format's speed relative to BF16 for the same matrix (above 1 = faster). Few tokens "
                "(decode) read every weight once and are limited by memory, so fewer bits per weight win; many "
                "tokens (prefill) are limited by arithmetic, where Blackwell's FP4 tensor cores matter. "
                + " ".join(f"{k}: {v}." for k, v in f["kernels"].items()), fig,
                raw=df.pivot_table(index=["matrix", "tokens"], columns="precision", values="ms")
                .mul(1000).round(1).to_html())
            fl = df.groupby("precision").first_ms.max().round(0)
            add("4c · First use vs steady state",
                "Slowest first call per format (kernel load or JIT compile) in milliseconds; steady-state calls are "
                "microseconds. A server pays this once per kernel variant at startup or on first use.",
                raw=fl.to_frame("first call ms (max)").to_html())
    head = ""
    if env:
        head = (f"{env['gpu']} (sm_{env['sm'].replace('.', '')}, {env['sms']} SMs) · FlashInfer {env['flashinfer']} · "
                f"torch {env['torch']} (CUDA {env['cuda']}) · Kimi K3 shapes: 24 MLA layers x 96 heads (latent 512 + "
                f"RoPE 64), routed experts 3584 -> 3072, TP8 across 8 GPUs.")
    if not parts:
        parts.append("<section><p class='why'>No results yet: run `python fi_bench_kimi.py mla` and `fp4`.</p></section>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
            f"initial-scale=1'><title>Kimi K3 kernels</title><style>{CSS}</style></head><body><main>"
            f"<h1>Kimi K3 on Blackwell: MLA attention and FP4 kernels</h1><p class='sub'>{html.escape(head)}</p>"
            + "".join(parts) + "</main></body></html>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fi_report.html").write_text(page)
    log(f"report: {OUT / 'fi_report.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mla", "fp4", "report"])
    ap.add_argument("--quick", action="store_true", help="one shape per test (pipeline check)")
    a = ap.parse_args()
    log(f"Kimi K3 Phase 4: {a.cmd} -> {OUT}")
    {"mla": cmd_mla, "fp4": cmd_fp4, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
