"""Qwen3.6-27B layer shapes from config.json, for the training load (train_proxy.py) and the report.

Input: the model's config.json (text_config is used when present). Output: per-layer linear shapes,
parameter counts and FLOPs per token, so the proxy matches the real model's compute and weight bytes.
"""
import json
from dataclasses import dataclass
from pathlib import Path

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")   # D6 scope


@dataclass(frozen=True)
class Shapes:
    hidden: int = 5120
    intermediate: int = 17408
    layers: int = 64
    heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    attn_gate: bool = True                 # q_proj also produces the output gate (2x width)
    full_every: int = 4                    # every 4th layer is full attention, the rest linear attention
    lin_k_heads: int = 16
    lin_v_heads: int = 48
    lin_k_dim: int = 128
    lin_v_dim: int = 128
    vocab: int = 248320

    @classmethod
    def from_config(cls, path) -> "Shapes":
        c = json.loads(Path(path).read_text())
        t = c.get("text_config", c)
        return cls(t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"], t["num_attention_heads"],
                   t["num_key_value_heads"], t["head_dim"], bool(t.get("attn_output_gate", False)),
                   t.get("full_attention_interval", 4), t["linear_num_key_heads"], t["linear_num_value_heads"],
                   t["linear_key_head_dim"], t["linear_value_head_dim"], t["vocab_size"])

    def is_full(self, i: int) -> bool:
        return (i + 1) % self.full_every == 0

    def linears(self, full: bool) -> dict[str, tuple[int, int]]:
        """name -> (in, out) of every linear layer in one decoder layer of this type."""
        h, a = self.hidden, self.heads * self.head_dim
        mlp = {"gate_proj": (h, self.intermediate), "up_proj": (h, self.intermediate),
               "down_proj": (self.intermediate, h)}
        if full:
            kv = self.kv_heads * self.head_dim
            return {"q_proj": (h, a * (2 if self.attn_gate else 1)), "k_proj": (h, kv), "v_proj": (h, kv),
                    "o_proj": (a, h), **mlp}
        v = self.lin_v_heads * self.lin_v_dim
        return {"in_proj_qkv": (h, 2 * self.lin_k_heads * self.lin_k_dim + v), "in_proj_z": (h, v),
                "in_proj_ab": (h, 2 * self.lin_v_heads), "out_proj": (v, h), **mlp}

    def layer_params(self, full: bool) -> int:
        return sum(i * o for i, o in self.linears(full).values())

    def lora_params(self, rank: int) -> int:
        return sum(rank * (i + o) for k in range(self.layers)
                   for n, (i, o) in self.linears(self.is_full(k)).items() if n in LORA_TARGETS)

    def matmul_params(self) -> int:
        """Weights that take part in a matmul per token: all decoder linears + the output layer."""
        return sum(self.layer_params(self.is_full(k)) for k in range(self.layers)) + self.hidden * self.vocab

    def weight_bytes(self, bytes_per_param: float = 2.0) -> float:
        """All weights incl. the input embedding (looked up, not multiplied)."""
        return (self.matmul_params() + self.hidden * self.vocab) * bytes_per_param

    def attn_flops_fwd(self, seq: int) -> float:
        """Causal attention FLOPs of one forward over `seq` tokens, all full-attention layers."""
        n_full = sum(self.is_full(k) for k in range(self.layers))
        return n_full * 2.0 * seq * seq * self.heads * self.head_dim      # QK^T + PV, causal half each

    def train_flops(self, seq: int) -> float:
        """One micro-batch as Miles runs it: old-log-prob forward + forward + recompute + backward.
        Linears: 2 + 2 + 2 + 2 (input grads only; LoRA weight grads are negligible) = 8 x params x tokens.
        Attention: 3 forwards + backward (~2.5 forwards)."""
        return 8.0 * self.matmul_params() * seq + 5.5 * self.attn_flops_fwd(seq)
