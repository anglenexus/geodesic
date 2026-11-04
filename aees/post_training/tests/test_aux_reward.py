"""grpo/aux_reward.patch (--opd-baseline exact) on the laptop: the code is rebuilt from the patch itself (the new
module aux_q.py, the added opd.py function), so no Miles checkout is needed. The Megatron passes run in
tests/gpu/aux_q_e2e.sh. Also configs/grpo.sh's AUX switch."""
import os
import re
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
PATCH = (ROOT / "grpo" / "aux_reward.patch").read_text()


def file_diff(path: str) -> str:
    m = re.search(rf"^diff --git a/{re.escape(path)} .*?(?=^diff --git |\Z)", PATCH, re.S | re.M)
    assert m, path
    return m.group(0)


def new_file(path: str) -> str:
    """A file the patch creates: its '+' lines."""
    body = file_diff(path).split("\n@@", 1)[1].split("\n", 1)[1]
    return "\n".join(line[1:] for line in body.splitlines() if line.startswith("+"))


def added_function(path: str, name: str) -> str:
    """One function the patch adds to an existing file (a contiguous run of '+' lines)."""
    added = "\n".join(line[1:] for line in file_diff(path).splitlines() if line.startswith("+") and not line.startswith("+++"))
    m = re.search(rf"^def {name}\(.*?(?=^def |^class |\Z)", added + "\n", re.S | re.M)
    assert m, name
    return m.group(0)


aux_q = type(os)("aux_q")
exec(new_file("miles/backends/megatron_utils/aux_q.py"), aux_q.__dict__)
ns = {"Namespace": Namespace, "torch": torch, "RolloutBatch": dict}
exec(added_function("miles/backends/training_utils/loss/hub/opd.py", "apply_opd_exact_to_advantages"), ns)
apply_exact = ns["apply_opd_exact_to_advantages"]


def test_patch_touches_only_the_expected_files_and_keeps_the_default():
    files = re.findall(r"^diff --git a/(\S+)", PATCH, re.M)
    assert files == ["miles/backends/megatron_utils/actor.py", "miles/backends/megatron_utils/aux_q.py",
                     "miles/backends/megatron_utils/model.py",
                     "miles/backends/training_utils/loss/hub/logit_processors.py",
                     "miles/backends/training_utils/loss/hub/opd.py", "miles/ray/specs/train.py",
                     "miles/utils/arguments.py"]
    args = file_diff("miles/utils/arguments.py")
    assert '+                choices=["student_logp", "exact"],' in args and '+                default="student_logp",' in args
    removed = re.findall(r"^-(?!--)(.*)$", PATCH, re.M)
    assert removed == ['                and trainer_args.opd_type == "megatron",',      # the teacher-model flag
                       '        if args.opd_type == "megatron":'], removed               # -> elif after the LoRA check
    assert '+        elif args.opd_type == "megatron":' in PATCH
    assert "+                and not trainer_args.opd_teacher_lora_path," in PATCH


# --- aux_q.aq_rows: A^Q = Q(s, x) − Σ_a π(a|s)·Q(s, a) ---

def setup(n=6, vocab=11, hidden=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, vocab, generator=g), torch.randn(n, hidden, generator=g),
            torch.randn(vocab, hidden, generator=g), torch.randint(0, vocab, (n,), generator=g))


def test_aq_rows_match_a_direct_computation():
    z_pi, h_q, w, x = setup()
    z_q = (h_q @ w.T).double()
    want = z_q[torch.arange(6), x] - (z_pi.double().softmax(-1) * z_q).sum(-1)
    torch.testing.assert_close(aux_q.aq_rows(z_pi, h_q, w, x).double(), want, rtol=1e-5, atol=1e-5)


def test_hand_example():
    """π = (1/2, 1/4, 1/4), Q = (2, 0, 4): V = Σπ·Q = 2, so A^Q = Q(x) − 2."""
    z_pi = torch.log(torch.tensor([[0.5, 0.25, 0.25]] * 3))
    got = aux_q.aq_rows(z_pi, torch.tensor([[2.0, 0.0, 4.0]] * 3), torch.eye(3), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(got, torch.tensor([0.0, -2.0, 2.0]))


def test_level_free_zero_mean_under_pi_and_matches_the_offline_definition():
    """A per-row constant added to Q cancels; Σ_x π(x)·A^Q(x) = 0; and A^Q = log p_Q(x) − Σ π·log p_Q (the form
    aees/reward/qfunc/aq.py computes offline)."""
    z_pi, h_q, w, x = setup()
    w1, shift = torch.cat([w, torch.ones(11, 1)], 1), torch.tensor([[0.0], [5.0], [-3.0], [40.0], [1.0], [2.0]])
    base = aux_q.aq_rows(z_pi, torch.cat([h_q, torch.zeros(6, 1)], 1), w1, x)
    torch.testing.assert_close(aux_q.aq_rows(z_pi, torch.cat([h_q, shift], 1), w1, x), base, rtol=1e-4, atol=1e-4)
    pi = z_pi.double().softmax(-1)
    every = torch.stack([aux_q.aq_rows(z_pi, h_q, w, torch.full((6,), v)) for v in range(11)], -1).double()
    torch.testing.assert_close((pi * every).sum(-1), torch.zeros(6, dtype=torch.float64), rtol=0, atol=1e-5)
    logp_q = (h_q @ w.T).double().log_softmax(-1)
    torch.testing.assert_close(base.double(), logp_q[torch.arange(6), x] - (pi * logp_q).sum(-1), rtol=1e-5, atol=1e-5)


def test_temperature_scales_pi_not_q_and_padded_vocab_is_dropped():
    z_pi, h_q, w, x = setup()
    z_q = (h_q @ w.T).double()
    hot = z_q[torch.arange(6), x] - ((z_pi.double() / 0.7).softmax(-1) * z_q).sum(-1)
    torch.testing.assert_close(aux_q.aq_rows(z_pi, h_q, w, x, temperature=0.7).double(), hot, rtol=1e-5, atol=1e-5)
    # Megatron pads the vocab: logits beyond args.vocab_size must not enter π or V
    padded_pi, padded_w = torch.cat([z_pi, torch.zeros(6, 5)], 1), torch.cat([w, torch.zeros(5, 5)], 0)
    torch.testing.assert_close(aux_q.aq_rows(padded_pi, h_q, padded_w, x, vocab=11), aux_q.aq_rows(z_pi, h_q, w, x))


@pytest.mark.parametrize("chunk", [1, 4, 1000])
def test_chunking_and_empty(chunk):
    z_pi, h_q, w, x = setup()
    torch.testing.assert_close(aux_q.aq_rows(z_pi, h_q, w, x, chunk=chunk), aux_q.aq_rows(z_pi, h_q, w, x))
    assert aux_q.aq_rows(z_pi[:0], h_q[:0], w, x[:0]).shape == (0,)


# --- aux_q.AuxQ: teacher pass saves hidden states per micro-batch, policy pass reads them in order ---

class Chunk(torch.nn.Module):
    """Stands in for a Megatron model chunk: a body and an output_layer fed [S, 1, H] (sequence first)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.body = torch.nn.Linear(4, 4)
        self.output_layer = torch.nn.Linear(4, 9, bias=False)
        with torch.no_grad():
            self.body.weight.copy_(torch.randn(4, 4, generator=g))
            self.output_layer.weight.copy_(torch.randn(9, 4, generator=torch.Generator().manual_seed(99)))

    def forward(self, x):                       # x [S, 4] -> logits [1, S, 9]
        return self.output_layer(self.body(x)[:, None]).transpose(0, 1)


def test_teacher_then_policy_pass_pair_micro_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_AUX_Q_DIR", str(tmp_path / "aux"))
    args = Namespace(opd_baseline="exact")
    teacher, policy = [Chunk(1)], [Chunk(2)]
    batches = [torch.randn(5, 4), torch.randn(3, 4)]
    aux = aux_q.begin(args, teacher, "teacher_")
    for b in batches:
        assert aux.after_forward(teacher[0](b)) == {}
    aux.end()
    aux = aux_q.begin(args, policy, "")
    for b in batches:
        out = aux.after_forward(policy[0](b))
        h_teacher = teacher[0].body(b)[None]
        torch.testing.assert_close(out["aux_hidden"], h_teacher)          # the same micro-batch's teacher states
        assert out["aux_weight"] is policy[0].output_layer.weight
    aux.end()
    assert not (tmp_path / "aux").exists()                                 # cleaned after the policy pass


class TiedChunk(torch.nn.Module):
    """Output layer without its own weight: Megatron passes the (tied) embedding weight at call time."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(9, 4, generator=torch.Generator().manual_seed(5)))
        self.output_layer = Head()

    def forward(self, x):
        return self.output_layer(x[:, None], weight=self.embed).transpose(0, 1)


class Head(torch.nn.Module):
    weight = None

    def forward(self, h, weight=None):
        return h @ weight.T


def test_tied_output_layer_uses_the_weight_passed_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_AUX_Q_DIR", str(tmp_path / "aux"))
    args, model, b = Namespace(opd_baseline="exact"), [TiedChunk()], torch.randn(5, 4)
    aux = aux_q.begin(args, model, "teacher_")
    aux.after_forward(model[0](b))
    aux.end()
    aux = aux_q.begin(args, model, "")
    out = aux.after_forward(model[0](b))
    aux.end()
    assert out["aux_weight"] is model[0].embed


def test_mismatched_passes_fail_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_AUX_Q_DIR", str(tmp_path / "aux"))
    args = Namespace(opd_baseline="exact")
    with pytest.raises(AssertionError, match="teacher pass first"):
        aux_q.begin(args, [Chunk(1)], "")
    teacher, policy = [Chunk(1)], [Chunk(2)]
    aux = aux_q.begin(args, teacher, "teacher_")
    aux.after_forward(teacher[0](torch.randn(5, 4)))
    aux.after_forward(teacher[0](torch.randn(3, 4)))
    aux.end()
    aux = aux_q.begin(args, policy, "")
    with pytest.raises(AssertionError, match="micro-batch 0"):
        aux.after_forward(policy[0](torch.randn(4, 4)))                    # different packing
    aux.hook.remove()
    aux = aux_q.begin(args, policy, "")
    aux.after_forward(policy[0](torch.randn(5, 4)))
    with pytest.raises(AssertionError, match="ran 1 micro-batches, teacher pass 2"):
        aux.end()


def test_begin_is_off_unless_exact_and_for_other_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_AUX_Q_DIR", str(tmp_path / "aux"))
    assert aux_q.begin(Namespace(opd_baseline="student_logp"), [Chunk(1)], "teacher_") is None
    assert aux_q.begin(Namespace(), [Chunk(1)], "") is None
    assert aux_q.begin(Namespace(opd_baseline="exact"), [Chunk(1)], "ref_") is None


# --- opd.apply_opd_exact_to_advantages ---

def rollout(n=3):
    g = torch.Generator().manual_seed(0)
    lengths = [4, 6, 2][:n]
    return {"aux_q": [torch.randn(k, generator=g) for k in lengths],
            "teacher_log_probs": [-torch.rand(k, generator=g) for k in lengths]}, \
        [torch.full((k,), float(i)) for i, k in enumerate(lengths)], [-torch.rand(k, generator=g) for k in lengths]


def test_exact_adds_beta_times_aq_and_logs_reverse_kl():
    data, adv, student = rollout()
    before = [a.clone() for a in adv]
    apply_exact(Namespace(opd_kl_coef=0.3), data, adv, student)
    for a, b, q in zip(adv, before, data["aux_q"]):
        torch.testing.assert_close(a, b + 0.3 * q)
    for kl, s, t in zip(data["opd_reverse_kl"], student, data["teacher_log_probs"]):
        torch.testing.assert_close(kl, s - t)


def test_beta_zero_is_vanilla_grpo():
    data, adv, student = rollout()
    before = [a.clone() for a in adv]
    apply_exact(Namespace(opd_kl_coef=0.0), data, adv, student)
    assert all(torch.equal(a, b) for a, b in zip(adv, before))


def test_exact_rejects_missing_or_misshapen_inputs():
    data, adv, student = rollout()
    with pytest.raises(ValueError, match="needs aux_q"):
        apply_exact(Namespace(opd_kl_coef=1.0), {"teacher_log_probs": data["teacher_log_probs"]}, adv, student)
    data["aux_q"][1] = torch.zeros(5)
    with pytest.raises(ValueError, match="shape mismatch at sample 1"):
        apply_exact(Namespace(opd_kl_coef=1.0), data, adv, student)
    data, adv, student = rollout()
    with pytest.raises(ValueError, match="length mismatch"):
        apply_exact(Namespace(opd_kl_coef=1.0), data, adv[:2], student)


# --- configs/grpo.sh ---

def grpo_args(**env) -> list[str]:
    r = subprocess.run(["bash", "-c", 'source configs/grpo.sh && printf "%s\\n" "${GRPO_ARGS[@]}"'], cwd=ROOT,
                       env={"PATH": os.environ["PATH"], "SAVE_DIR": "/tmp/x", "MAX_LEN": "49152", **env},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_aux_switch_adds_the_teacher_args_and_default_is_unchanged():
    base = grpo_args()
    assert not any(a.startswith("--opd") or a == "--use-opd" for a in base)
    aux = grpo_args(AUX="1", AUX_BETA="0.25")
    want = ["--use-opd", "--opd-type", "megatron", "--opd-teacher-lora-path", "/root/models/qfunc_adapter",
            "--opd-kl-coef", "0.25", "--opd-baseline", "exact"]
    i = aux.index("--use-opd")
    assert aux[i:i + len(want)] == want and aux[:i] + aux[i + len(want):] == base     # inserted, nothing else changed


def test_aux_without_beta_fails():
    r = subprocess.run(["bash", "-c", "source configs/grpo.sh"], cwd=ROOT,
                       env={"PATH": os.environ["PATH"], "SAVE_DIR": "/tmp/x", "MAX_LEN": "49152", "AUX": "1"},
                       capture_output=True, text=True)
    assert r.returncode != 0 and "AUX_BETA" in r.stderr


# --- aux_q.teacher_lora: the teacher's adapter swapped in for the teacher pass, the policy's restored after ---

def test_teacher_lora_swaps_in_and_restores_even_on_error(monkeypatch):
    import sys, types
    params = {"a": torch.nn.Parameter(torch.ones(3)), "b": torch.nn.Parameter(torch.zeros(2))}
    loaded = []

    def fake_load(model, path):
        loaded.append(path)
        with torch.no_grad():
            params["a"].fill_(7.0); params["b"].fill_(9.0)

    mods = {"miles": types.ModuleType("miles"),
            "miles.backends.megatron_utils.lora.native.hf_adapter": types.SimpleNamespace(load_lora_adapter_hf=fake_load),
            "miles.backends.megatron_utils.lora.utils": types.SimpleNamespace(_adapter_shard_params=lambda m: params)}
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    with aux_q.teacher_lora("model", "/q/adapter"):
        assert params["a"].tolist() == [7.0] * 3 and params["b"].tolist() == [9.0] * 2
    assert params["a"].tolist() == [1.0] * 3 and params["b"].tolist() == [0.0] * 2 and loaded == ["/q/adapter"]
    with pytest.raises(RuntimeError):
        with aux_q.teacher_lora("model", "/q/adapter"):
            raise RuntimeError("teacher pass failed")
    assert params["a"].tolist() == [1.0] * 3
