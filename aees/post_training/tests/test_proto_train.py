"""Tests for proto/shapes.py, proto/train_proxy.py (on CPU, tiny shapes) and proto/adapter.py."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from proto import adapter, train_proxy as tp
from proto.shapes import LORA_TARGETS, Shapes

REAL = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {   # Qwen/Qwen3.6-27B config.json
    "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64, "num_attention_heads": 24,
    "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True, "full_attention_interval": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "vocab_size": 248320}}
TINY = {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 8, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "attn_output_gate": True, "full_attention_interval": 4,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8, "vocab_size": 97}
CPU, F32 = torch.device("cpu"), torch.float32


def cfg(tmp_path, c):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    return str(p)


def test_real_shapes_match_the_model(tmp_path):
    s = Shapes.from_config(cfg(tmp_path, REAL))
    assert s == Shapes()                                               # defaults are Qwen3.6-27B
    assert 26.8e9 < s.weight_bytes(1) < 26.95e9                        # 26.9B params (D1)
    assert s.lora_params(64) == 318_767_104                            # 318.7M (D6)
    assert [k for k in range(64) if s.is_full(k)] == list(range(3, 64, 4))
    assert s.linears(True)["q_proj"] == (5120, 12288)                  # query + output gate
    assert s.linears(False)["in_proj_qkv"] == (5120, 10240)
    assert s.train_flops(1000) == 8 * s.matmul_params() * 1000 + 5.5 * s.attn_flops_fwd(1000)
    assert s.attn_flops_fwd(2000) == pytest.approx(4 * s.attn_flops_fwd(1000))


def tiny_model(tmp_path, physical=4):
    s = Shapes.from_config(cfg(tmp_path, TINY))
    torch.manual_seed(0)
    return s, tp.Proxy(s, physical, 4, CPU, F32)


def lora_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def test_logprobs_match_log_softmax_chunked_or_full(tmp_path):
    s, m = tiny_model(tmp_path)
    x, t = torch.randn(40, s.hidden), torch.randint(0, s.vocab, (40,))
    h = m.hidden(x, recompute=False)
    ref = torch.log_softmax(torch.nn.functional.linear(h, m.head.w), -1).gather(-1, t[:, None])[:, 0]
    for chunk in (7, 40, 64):
        for full in (False, True):
            assert torch.allclose(m.logprobs(h, t, chunk, full), ref, atol=1e-5)


@pytest.mark.parametrize("variant", ["recompute", "chunked", "offload"])
def test_recompute_chunking_offload_give_the_same_gradients(tmp_path, variant):
    s, m = tiny_model(tmp_path)
    x, t = torch.randn(24, s.hidden), torch.randint(0, s.vocab, (24,))
    m.logprobs(m.hidden(x, False), t, 24, True).sum().backward()
    ref = lora_grads(m)
    m.zero_grad()
    ctx = torch.autograd.graph.save_on_cpu(pin_memory=False) if variant == "offload" else torch.enable_grad()
    with ctx:
        lp = m.logprobs(m.hidden(x, variant != "chunked"), t, 5 if variant == "chunked" else 24, False)
    lp.sum().backward()
    got = lora_grads(m)
    assert ref.keys() == got.keys() and len(ref) == 2 * (7 + 3 * 3)   # A, B: 1 full layer (4 attn + 3 MLP), 3 linear (MLP)
    for n in ref:
        assert torch.allclose(ref[n], got[n], atol=1e-5), n


def test_only_lora_weights_train(tmp_path):
    s, m = tiny_model(tmp_path)
    base = {n: p.clone() for n, p in m.named_parameters() if not p.requires_grad}
    lora = {n: p.clone() for n, p in m.named_parameters() if p.requires_grad}
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    args = type("A", (), {"chunk": 8, "offload": False, "logits": "chunked"})
    tp.micro_batch(m, opt, 16, args, CPU, F32)
    assert all(torch.equal(base[n], p) for n, p in m.named_parameters() if n in base)
    assert all(not torch.equal(lora[n], p) for n, p in m.named_parameters() if n in lora)
    assert {n.rsplit(".", 2)[-2] for n in lora} <= set(LORA_TARGETS)


def test_physical_layers_must_keep_the_pattern(tmp_path):
    s = Shapes.from_config(cfg(tmp_path, TINY))
    with pytest.raises(AssertionError):
        tp.Proxy(s, 3, 4, CPU, F32)


@pytest.mark.parametrize("extra", [[], ["--offload"], ["--logits", "full"], ["--offload", "--logits", "full"]])
def test_main_writes_start_and_step_rows(tmp_path, extra):
    out = tmp_path / "train.jsonl"
    rc = tp.main(["--config", cfg(tmp_path, TINY), "--out", str(out), "--seq", "32", "--max-steps", "3",
                  "--physical", "4", "--chunk", "8", "--device", "cpu", "--dtype", "float32", *extra])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    s = Shapes.from_config(cfg(tmp_path, TINY))
    assert rc == 0 and [r["type"] for r in rows] == ["start", "step", "step", "step"]
    assert rows[0]["flops_per_mb"] == s.train_flops(32) and rows[0]["offload"] == ("--offload" in extra)
    assert rows[0]["ballast_gb"] == round(max(0, s.weight_bytes() - 4 * sum(
        i * o for k in range(4) for i, o in s.linears(s.is_full(k)).values()) - 4 * s.hidden * s.vocab) / 2 ** 30, 1)
    assert all(r["tflops"] >= 0 and r["seconds"] > 0 and r["tokens"] == 32 for r in rows[1:])   # tiny: ~0 TFLOPS


def test_main_reports_out_of_memory(tmp_path, monkeypatch):
    def boom(*a):
        raise torch.OutOfMemoryError("CUDA out of memory")
    monkeypatch.setattr(tp, "micro_batch", boom)
    out = tmp_path / "t.jsonl"
    rc = tp.main(["--config", cfg(tmp_path, TINY), "--out", str(out), "--physical", "4", "--device", "cpu",
                  "--dtype", "float32", "--ballast", "0"])
    assert rc == 3 and json.loads(out.read_text().splitlines()[-1])["type"] == "oom"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_pcie():
    r = tp.pcie(torch.device("cuda"), gb=0.25, reps=1)
    assert r["h2d_gbps"] > 1 and r["d2h_gbps"] > 1


def fake_checkpoint(d):
    """Two shards with the real naming: decoder targets, a linear-attention projection, vision, MTP, head."""
    names = {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (40, 16),
             "model.language_model.layers.0.mlp.gate_proj.weight": (24, 16),
             "model.language_model.layers.0.mlp.down_proj.weight": (16, 24),
             "model.language_model.layers.3.self_attn.q_proj.weight": (32, 16),
             "model.language_model.layers.3.self_attn.k_proj.weight": (8, 16),
             "model.language_model.layers.3.self_attn.q_norm.weight": (8,),
             "model.visual.blocks.0.mlp.gate_proj.weight": (12, 6),
             "mtp.layers.0.mlp.up_proj.weight": (24, 16), "lm_head.weight": (50, 16)}
    shards = {}
    for i, (n, shape) in enumerate(names.items()):
        shards.setdefault(f"model-0000{i % 2 + 1}-of-00002.safetensors", {})[n] = torch.zeros(shape)
    for f, t in shards.items():
        save_file(t, str(d / f))
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: f for f, t in shards.items() for n in t}}))
    return names


def test_adapter_targets_decoder_linears_only(tmp_path):
    names = fake_checkpoint(tmp_path)
    assert adapter.weight_shapes(tmp_path) == {n: list(s) for n, s in names.items()}
    mods = adapter.lora_modules(adapter.weight_shapes(tmp_path))
    assert mods == {"model.language_model.layers.0.mlp.gate_proj": (16, 24),
                    "model.language_model.layers.0.mlp.down_proj": (24, 16),
                    "model.language_model.layers.3.self_attn.q_proj": (16, 32),
                    "model.language_model.layers.3.self_attn.k_proj": (16, 8)}
    n = adapter.write_adapter(tmp_path, tmp_path / "ad", rank=4)
    t = load_file(str(tmp_path / "ad" / "adapter_model.safetensors"))
    assert n == sum(4 * (i + o) for i, o in mods.values()) == sum(v.numel() for v in t.values())
    for m, (i, o) in mods.items():
        a, b = t[f"base_model.model.{m}.lora_A.weight"], t[f"base_model.model.{m}.lora_B.weight"]
        assert a.shape == (4, i) and b.shape == (o, 4) and not b.any() and a.dtype == torch.bfloat16
    c = json.loads((tmp_path / "ad" / "adapter_config.json").read_text())
    assert c["r"] == c["lora_alpha"] == 4 and c["peft_type"] == "LORA"
    assert c["target_modules"] == ["down_proj", "gate_proj", "k_proj", "q_proj"]
