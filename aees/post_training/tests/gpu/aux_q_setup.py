"""GPU test of --opd-baseline exact (grpo/aux_reward.patch), inside the Miles image: inputs.

    python3 aux_q_setup.py teacher <base HF dir> <out dir>     # base + a random perturbation of the LoRA-target
                                                               # projections (as a merged adapter); the output layer,
                                                               # embeddings and norms stay identical
    python3 aux_q_setup.py lora <base HF dir> <out dir>        # a PEFT LoRA (rank 8, the 7 projections) made like
                                                               # the Q adapter (PEFT on the text model), B != 0;
                                                               # <out dir>/merged: base + it, for the reference
    python3 aux_q_setup.py rollout <base HF dir> <out .pt>     # 2 groups x 3 samples: prompt, then replies (loss 1)
                                                               # and observations (loss 0), mixed rewards
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def teacher(base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    changed = 0
    for f in base.iterdir():
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy(f, out / f.name)
            continue
        t = load_file(f)
        for k, v in t.items():
            if any(f".{p}." in k for p in TARGETS) and "layers." in k and "visual" not in k and "mtp" not in k:
                d = 0.05 * v.float().std() * torch.randn(v.shape, generator=g)
                t[k] = (v.float() + d).to(v.dtype)
                changed += 1
        save_file(t, out / f.name, metadata={"format": "pt"})
    print(f"TEACHER_OK changed={changed}")


def lora(base: Path, out: Path) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(base).get_text_config()
    model = AutoModelForCausalLM.from_pretrained(base, config=cfg, dtype=torch.float32)
    torch.manual_seed(0)
    peft = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=list(TARGETS),
                                            bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02)
    peft.save_pretrained(out / "peft")                          # text-model names, as the Q adapter has
    import subprocess
    subprocess.run([sys.executable, "/root/post_training/grpo/q_adapter.py", str(out / "peft"), str(out)], check=True)
    merged = peft.merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(out / "merged")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if (base / f).exists():
            shutil.copy(base / f, out / "merged" / f)
    print(f"LORA_OK {out}")


def rollout(base: Path, out: Path) -> None:
    from miles.utils.types import Sample
    vocab = json.loads((base / "config.json").read_text()).get("text_config", {}).get("vocab_size") or 151000
    g = torch.Generator().manual_seed(1)
    samples, rewards = [], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # group 0 mixed, group 1 all-fail
    for i in range(6):
        prompt = torch.randint(100, 20000, (60,), generator=g).tolist()
        response, mask = [], []
        for turn in range(3):                                         # reply (trained), observation (not)
            n_reply = 40 + 10 * turn + i
            response += torch.randint(100, 20000, (n_reply,), generator=g).tolist(); mask += [1] * n_reply
            if turn < 2:
                n_obs = 25
                response += torch.randint(100, 20000, (n_obs,), generator=g).tolist(); mask += [0] * n_obs
        s = Sample(group_index=i // 3, index=i, prompt="x", tokens=prompt + response, response_length=len(response),
                   reward=rewards[i], loss_mask=mask, rollout_log_probs=[0.0] * len(response),
                   status=Sample.Status.COMPLETED)
        samples.append(s.to_dict())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rollout_id=0, metadata={}, samples=samples), out)
    print(f"ROLLOUT_OK samples={len(samples)} vocab={vocab}")


if __name__ == "__main__":
    {"teacher": teacher, "lora": lora, "rollout": rollout}[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
