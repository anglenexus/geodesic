"""GPU test of --opd-baseline exact: checks Miles' dumped train data against an independent HF computation.

    python3 aux_q_check.py <train dump .pt> <rollout .pt> <base HF dir> <teacher HF dir> <coef>

Reference, per sample and response position t (target x = token t+1): A^Q = Q(s_t, x) − Σ_a softmax(z_π)(a)·Q(s_t, a),
Q = the teacher's raw logits, z_π = the base model's (the policy at step 0: LoRA B = 0). Checks Miles' aux_q against
it, and advantages − GRPO term = coef·aux_q.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load(path):
    cfg = AutoConfig.from_pretrained(path)
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16).cuda().eval()


def grpo(rewards, groups):
    out = torch.zeros(len(rewards))
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        r = torch.tensor([rewards[i] for i in idx])
        a = r - r.mean()
        if len(r) > 1 and r.std() > 0:
            a = a / (r.std() + 1e-6)
        out[idx] = a
    return out


def main(dump, roll, base, teacher, coef):
    d = torch.load(dump, weights_only=False)
    data = d.get("rollout_data", d)
    print("dump keys:", sorted(k for k in data if not k.startswith("_"))[:40])
    samples = torch.load(roll, weights_only=False)["samples"]
    pi_model, q_model = load(base), load(teacher)
    assert torch.equal(pi_model.lm_head.weight, q_model.lm_head.weight), "the teacher must share the output layer"
    g = grpo([s["reward"] for s in samples], [s["group_index"] for s in samples])
    worst, corr_in = 0.0, []
    order = [int(x) for x in data["sample_indices"]]               # Miles' row j holds rollout sample order[j]
    assert sorted(order) == list(range(len(samples))), order
    for j, i in enumerate(order):
        s = samples[i]
        ids = torch.tensor(s["tokens"], device="cuda")[None]
        R, T = s["response_length"], len(s["tokens"])
        with torch.no_grad():
            zp, zq = pi_model(input_ids=ids).logits[0].float(), q_model(input_ids=ids).logits[0].float()
        rows = torch.arange(T - R - 1, T - 1, device="cuda")
        x = ids[0, rows + 1]
        ref = zq[rows].gather(-1, x[:, None]).squeeze(-1) - (zp[rows].softmax(-1) * zq[rows]).sum(-1)
        got = torch.as_tensor(data["aux_q"][j]).float().cuda()
        assert got.shape == ref.shape, (got.shape, ref.shape)
        err = (got - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-3)
        c = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
        worst = max(worst, err); corr_in.append(c)
        adv = torch.as_tensor(data["advantages"][j]).float().cuda()
        diff = (adv - g[i].item() - coef * got).abs().max().item()
        print(f"sample {i} (row {j}): R={R} rel_err={err:.4f} corr={c:.5f} |adv - grpo - coef*aux_q|max={diff:.2e} "
              f"aux_q mean={got.mean():.3f} std={got.std():.3f}")
        assert diff < 1e-3, "advantages != GRPO + coef·A^Q"
    assert worst < 0.05 and min(corr_in) > 0.99, (worst, min(corr_in))
    print(f"AUX_Q_E2E_OK worst_rel_err={worst:.4f} min_corr={min(corr_in):.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
