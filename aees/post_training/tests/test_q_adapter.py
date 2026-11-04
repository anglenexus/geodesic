"""grpo/q_adapter.py: the Q adapter's keys renamed to Miles' layout, nothing else changed."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grpo"))
import q_adapter  # noqa: E402


def test_renames_text_model_keys_and_keeps_tensors(tmp_path):
    t = {"base_model.model.model.layers.3.mlp.gate_proj.lora_A.weight": torch.randn(4, 8),
         "base_model.model.model.layers.3.mlp.gate_proj.lora_B.weight": torch.randn(16, 4),
         "base_model.model.model.layers.15.self_attn.q_proj.lora_A.weight": torch.randn(4, 8)}
    (tmp_path / "src").mkdir()
    save_file(t, tmp_path / "src" / "adapter_model.safetensors")
    (tmp_path / "src" / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 4}))
    assert q_adapter.convert(tmp_path / "src", tmp_path / "out") == 3
    got = load_file(tmp_path / "out" / "adapter_model.safetensors")
    assert set(got) == {k.replace("model.model.layers.", "model.model.language_model.layers.") for k in t}
    for k, v in t.items():
        assert torch.equal(got[k.replace("model.model.layers.", "model.model.language_model.layers.")], v)
    assert json.loads((tmp_path / "out" / "adapter_config.json").read_text()) == {"r": 4, "lora_alpha": 4}
    assert q_adapter.convert(tmp_path / "out", tmp_path / "again") == 3          # already in Miles' layout: unchanged


def test_rejects_unexpected_keys(tmp_path):
    (tmp_path / "src").mkdir()
    save_file({"base_model.model.lm_head.lora_A.weight": torch.zeros(1)}, tmp_path / "src" / "adapter_model.safetensors")
    (tmp_path / "src" / "adapter_config.json").write_text("{}")
    with pytest.raises(AssertionError):
        q_adapter.convert(tmp_path / "src", tmp_path / "out")
