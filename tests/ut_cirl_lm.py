"""End-to-end smoke test for ContextIRLOrchestrator on Shakespeare BPE data.

Data is prepared by data/lm/prepare.py (downloads tiny shakespeare and tokenizes
with GPT-2 BPE). The test auto-runs prepare.py if train.bin is missing.

Surrogates are sampled uniformly across the vocabulary by default; the unigram
counts are still computed (exercising the compute_tf_counts pipeline) but not
used for sampling here. Asserts that loss drops below the random-chance
baseline log(K).
"""

import math
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext

import torch

from cirl_lm_orchestrator import (
    RewardGPT, RewardGPTConfig,
    ContextIRLReward, ContextIRLOrchestrator,
    get_batch, compute_tf_counts, estimate_loss, get_lr, configure_optimizers,
)

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'lm')
)
VOCAB_SIZE = 50304   # nanoGPT convention; GPT-2 BPE actual vocab is 50257


def ensure_data():
    if os.path.exists(os.path.join(DATA_DIR, 'train.bin')):
        return
    print(f"train.bin missing in {DATA_DIR} — running prepare.py...")
    subprocess.run(
        [sys.executable, os.path.join(DATA_DIR, 'prepare.py')],
        check=True,
    )


def main():
    ensure_data()
    torch.manual_seed(0)
    device, device_type = 'cpu', 'cpu'
    V = VOCAB_SIZE
    N, B = 64, 8
    K, D, gamma = 8, 10, 0.99
    out_dir = tempfile.mkdtemp(prefix='cirl_ut_')

    config = {
        'out_dir': out_dir,
        'max_iters': 500,
        'eval_interval': 50,
        'eval_iters': 5,
        'log_interval': 20,
        'eval_only': False,
        'always_save_checkpoint': True,
        'decay_lr': False,
        'learning_rate': 3e-3,
        'warmup_iters': 0,
        'lr_decay_iters': 200,
        'min_lr': 3e-5,
        'grad_clip': 1.0,
    }

    # ---- model + IRL wrapper ----
    reward_conf = RewardGPTConfig(
        block_size=N, vocab_size=V, n_layer=2, n_head=2, n_embd=32,
    )
    reward_net = RewardGPT(reward_conf).to(device)
    tf_counts = compute_tf_counts(DATA_DIR, V)   # computed but not used in 'uniform' mode
    irl = ContextIRLReward(
        reward_net, V, K=K, D=D, gamma=gamma,
        surrogate_mode='uniform',
    ).to(device)

    # ---- optimizer / scaler / ctx ----
    optimizer = configure_optimizers(
        reward_net, weight_decay=0.1, learning_rate=config['learning_rate'],
        betas=(0.9, 0.95), device_type=device_type,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    ctx = nullcontext()

    # ---- data + wired helpers ----
    def get_batch_fn(split):
        return get_batch(split, DATA_DIR, N, B, device, device_type)

    def estimate_loss_fn():
        return estimate_loss(irl, get_batch_fn, ctx, config['eval_iters'])

    def get_lr_fn(it):
        return get_lr(it, config['warmup_iters'], config['learning_rate'],
                      config['lr_decay_iters'], config['min_lr'])

    log_K = math.log(K)
    initial = estimate_loss_fn()
    print(f"initial train loss {initial['train_loss']:.4f} "
          f"(random-chance baseline log(K)={log_K:.3f})")

    ContextIRLOrchestrator(
        irl=irl, reward_net=reward_net, reward_conf=reward_conf,
        optimizer=optimizer, scaler=scaler, ctx=ctx,
        get_batch=get_batch_fn,
        estimate_loss=estimate_loss_fn, get_lr=get_lr_fn,
        config=config,
    ).start()

    final = estimate_loss_fn()
    print(f"final train loss {final['train_loss']:.4f}")

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    assert os.path.exists(ckpt_path), f"checkpoint missing at {ckpt_path}"

    assert final['train_loss'] < log_K * 0.95, (
        f"loss did not drop below random-chance baseline log(K)={log_K:.3f}: "
        f"got {final['train_loss']:.3f}"
    )
    print(f"OK — loss dropped from {initial['train_loss']:.3f} → "
          f"{final['train_loss']:.3f} (below baseline {log_K:.3f})")


if __name__ == '__main__':
    main()
