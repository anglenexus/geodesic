"""The Q-function's PEFT adapter in the key layout Miles' native LoRA loads (--opd-teacher-lora-path).

    python3 grpo/q_adapter.py <PEFT adapter dir> <out dir>

PEFT on the text model (aees/reward: Qwen3_5ForCausalLM) names modules base_model.model.model.layers.N...; Miles
(lora/native/hf_adapter.py) expects the image-text checkpoint's names, model.language_model.layers.N... after it
strips base_model.model. Only that prefix changes; tensors and adapter_config.json are copied as they are.
"""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

TEXT, MILES = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def convert(src: Path, out: Path) -> int:
    tensors = load_file(src / "adapter_model.safetensors")
    assert all(k.startswith((TEXT, MILES)) for k in tensors), [k for k in tensors if not k.startswith((TEXT, MILES))][:3]
    renamed = {(MILES + k[len(TEXT):] if k.startswith(TEXT) else k): v for k, v in tensors.items()}
    assert len(renamed) == len(tensors)
    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, out / "adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(src / "adapter_config.json", out / "adapter_config.json")
    return len(renamed)


if __name__ == "__main__":
    print(f"Q_ADAPTER_OK tensors={convert(Path(sys.argv[1]), Path(sys.argv[2]))}")
