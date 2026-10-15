"""End-to-end smoke test for ActorFromCIRLOrchestrator on synthetic data.

Trains an LMPolicyCritic against random per-position rewards in [0, 1] via the
DQN-style TD loss in ActorFromCIRLOrchestrator. Asserts the val TD loss drops
between pre-training and post-training evaluation — a plumbing check that
the orchestrator wires together end-to-end.

This is the actor-side training-loop smoke test. It deliberately does NOT
involve a trained reward model — the combined CIRL+actor integration test
(meaningful rewards + text-generation eval) is tracked separately.
"""

import os
import tempfile
from contextlib import nullcontext

import torch

from cirl_lm_orchestrator import (
    ActorFromCIRLOrchestrator, configure_optimizers, get_lr,
)
from lm_models import LMPolicyCritic, LMPolicyCriticConfig


def main():
    torch.manual_seed(0)
    device, device_type = 'cpu', 'cpu'
    V, N, B = 257, 32, 4                            # small smoke-test sizes
    out_dir = tempfile.mkdtemp(prefix='cirl_actor_ut_')

    config = {
        'out_dir': out_dir,
        'max_iters': 30,
        'eval_interval': 5,
        'eval_iters': 2,
        'log_interval': 5,
        'eval_only': False,
        'decay_lr': False,
        'learning_rate': 3e-4,
        'warmup_iters': 0,
        'lr_decay_iters': 30,
        'min_lr': 3e-5,
        'grad_clip': 1.0,
        'gamma': 0.99,
        # actor-specific (defaults are fine but listing explicitly for visibility):
        'use_target_net': False,
        'tau': 0.005,
        'bootstrap_mode': 'argmax',
        'bootstrap_temperature': 1.0,
    }

    # --- model + IRL wrapper ---
    model_cfg = LMPolicyCriticConfig(
        block_size=N, vocab_size=V, n_layer=2, n_head=2, n_embd=32,
    )                                          # is_causal=True by default
    policy_critic = LMPolicyCritic(model_cfg).to(device)

    # --- optimizer / scaler / autocast ctx ---
    optimizer = configure_optimizers(
        policy_critic, weight_decay=0.1, learning_rate=config['learning_rate'],
        betas=(0.9, 0.95), device_type=device_type,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    autocast_ctx = nullcontext()

    # --- synthetic data + wired helpers ---
    def synthetic_get_batch(split):
        tokens  = torch.randint(0, V, (B, N), device=device)
        rewards = torch.rand(B, N, device=device)          # (B, N), uniform in [0, 1]
        return tokens, rewards

    def get_lr_fn(it):
        return get_lr(it, config['warmup_iters'], config['learning_rate'],
                      config['lr_decay_iters'], config['min_lr'])

    # --- construct orchestrator, measure baseline, train, measure final ---
    orch = ActorFromCIRLOrchestrator(
        policy_critic=policy_critic,
        optimizer=optimizer,
        scaler=scaler,
        autocast_ctx=autocast_ctx,
        get_batch=synthetic_get_batch,
        get_lr=get_lr_fn,
        config=config,
    )

    initial = orch._evaluate()
    print(f"initial val loss {initial['val_loss']:.4f}  Q {initial['val_Q']:.3f}")
    orch.start()
    final = orch._evaluate()
    print(f"final   val loss {final['val_loss']:.4f}  Q {final['val_Q']:.3f}")

    # --- assertions ---
    assert final['val_loss'] < initial['val_loss'], (
        f"TD loss did not drop: initial {initial['val_loss']:.4f} -> "
        f"final {final['val_loss']:.4f}"
    )
    best_ckpt = os.path.join(out_dir, 'ckpt_best.pt')
    assert os.path.exists(best_ckpt), f"ckpt_best.pt missing at {best_ckpt}"

    print(f"OK - loss dropped from {initial['val_loss']:.4f} -> {final['val_loss']:.4f}")


if __name__ == '__main__':
    main()
