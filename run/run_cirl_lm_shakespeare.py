"""1-2 epoch training run of ContextIRLOrchestrator on Shakespeare BPE data.

Loads tokenized Shakespeare from data/lm (auto-runs prepare.py if missing).
Trains a RewardGPT for ~2 epochs with linear-warmup + cosine LR decay and
val-loss-driven checkpointing. Surrogates are sampled uniformly across
vocab; unigram counts are still computed (pipeline exercise) but not used.
Asserts final loss drops well below the random-chance baseline log(K).
"""

import math
import os
import subprocess
import sys
from contextlib import nullcontext

import torch

from cirl_lm_orchestrator import (
    RewardGPT, RewardGPTConfig,
    ContextIRLReward, ContextIRLOrchestrator,
    get_batch, compute_tf_counts, estimate_loss, get_lr, configure_optimizers,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR  = os.path.join(REPO_ROOT, 'data', 'lm')
CKPT_DIR  = os.path.join(REPO_ROOT, 'ckpts', 'cirl_lm_shakespeare')
VOCAB_SIZE = 50304   # nanoGPT convention; GPT-2 BPE actual vocab is 50257
TRAIN_TOKENS = 301_966   # produced by data/lm/prepare.py


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
    os.makedirs(CKPT_DIR, exist_ok=True)
    out_dir = CKPT_DIR

    iters_per_epoch = TRAIN_TOKENS // (B * N)
    epochs = 2
    max_iters = epochs * iters_per_epoch

    config = {
        'out_dir': out_dir,
        'max_iters': max_iters,
        'iters_per_epoch': iters_per_epoch,
        'eval_interval': 100,
        'eval_iters': 20,
        'log_interval': 50,
        'eval_only': False,
        'always_save_checkpoint': False,    # save only when val improves
        'decay_lr': True,
        'learning_rate': 1e-3,
        'warmup_iters': 100,
        'lr_decay_iters': max_iters,
        'min_lr': 1e-4,
        'grad_clip': 1.0,
    }

    print(f"data_dir={DATA_DIR}  V={V}  iters_per_epoch={iters_per_epoch}  "
          f"max_iters={max_iters} ({epochs} epochs)")

    # ---- model + IRL wrapper ----
    reward_conf = RewardGPTConfig(
        block_size=N, vocab_size=V, n_layer=2, n_head=4, n_embd=64,
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
    print(f"final train loss {final['train_loss']:.4f} acc {final['train_acc']:.3f}  |  "
          f"val loss {final['val_loss']:.4f} acc {final['val_acc']:.3f}")

    ckpt_path = os.path.join(out_dir, f'ckpt_epoch{epochs}.pt')
    assert os.path.exists(ckpt_path), f"final-epoch checkpoint missing at {ckpt_path}"

    assert final['train_loss'] < log_K * 0.5, (
        f"loss did not converge below 50% of random-chance baseline log(K)={log_K:.3f}: "
        f"got {final['train_loss']:.3f}"
    )
    print(f"OK — loss dropped from {initial['train_loss']:.3f} → "
          f"{final['train_loss']:.3f} (below 0.5 * baseline {log_K:.3f})")


if __name__ == '__main__':
    main()
