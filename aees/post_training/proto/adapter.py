"""Random LoRA adapter in PEFT format, named after the real model's modules, to time SGLang's adapter
hot-load (the per-step weight push of an overlapped design) and to serve rollouts through LoRA kernels.

  python -m proto.adapter /root/models/Qwen3.6-27B /root/proto_out/adapter --rank 64
Input: a model folder with model.safetensors.index.json (only the shard headers are read).
Output: adapter_config.json + adapter_model.safetensors. lora_B is zero, so outputs equal the base model's.
"""
import argparse
import json
import struct
from pathlib import Path

from proto.shapes import LORA_TARGETS


def weight_shapes(model_dir) -> dict[str, list[int]]:
    """Tensor name -> shape for every weight in the checkpoint, from the safetensors headers."""
    d = Path(model_dir)
    shards = sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values()))
    shapes = {}
    for shard in shards:
        with open(d / shard, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        shapes.update({k: v["shape"] for k, v in header.items() if k != "__metadata__"})
    return shapes


def lora_modules(shapes: dict, targets=LORA_TARGETS) -> dict[str, tuple[int, int]]:
    """Module path -> (in, out) for the decoder's target linears (no vision tower, no MTP head)."""
    out = {}
    for name, shape in shapes.items():
        mod, _, leaf = name.rpartition(".")
        if leaf == "weight" and mod.rpartition(".")[2] in targets and ".layers." in mod \
                and "visual" not in mod and not mod.startswith("mtp"):
            out[mod] = (shape[1], shape[0])
    return out


def write_adapter(model_dir, out_dir, rank: int = 64) -> int:
    """Writes the adapter; returns its parameter count."""
    import torch
    from safetensors.torch import save_file
    mods = lora_modules(weight_shapes(model_dir))
    assert mods, f"no {LORA_TARGETS} weights found under {model_dir}"
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for m, (i, o) in sorted(mods.items()):
        tensors[f"base_model.model.{m}.lora_A.weight"] = (torch.randn(rank, i, generator=g) * i ** -0.5).bfloat16()
        tensors[f"base_model.model.{m}.lora_B.weight"] = torch.zeros(o, rank, dtype=torch.bfloat16)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
        "bias": "none", "base_model_name_or_path": str(model_dir),
        "target_modules": sorted({m.rpartition(".")[2] for m in mods})}, indent=1))
    return sum(t.numel() for t in tensors.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("out_dir")
    p.add_argument("--rank", type=int, default=64)
    a = p.parse_args()
    print(f"adapter: {write_adapter(a.model_dir, a.out_dir, a.rank) / 1e6:.1f}M params -> {a.out_dir}")
