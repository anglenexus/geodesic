"""Integration test: CIRL reward model -> pre-cached rewards -> actor -> text generation.

Three-stage end-to-end pipeline on Shakespeare BPE data:
    Stage 1   train a small RewardGPT via ContextIRLOrchestrator      (~30-60 s CPU)
    Stage 1.5 call data.lm.prepare_rewards.prepare() for train + val  (~10-20 s CPU)
    Stage 2   train a small LMPolicyCritic via ActorFromCIRLOrchestrator,
              consuming pre-cached (tokens, rewards) batches          (~30-60 s CPU)
    Stage 3   load the actor's best ckpt, generate from a prompt,
              decode with GPT-2 BPE, print to stdout

Designed lightweight enough for CI (~2-3 min CPU total).

IT_SKIP_CIRL=1 reuses prior CIRL artifacts and runs only Stages 2 + 3.
"""

import math
import os
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
import tiktoken

from cirl_lm_orchestrator import (
    RewardGPT, RewardGPTConfig,
    ContextIRLReward, ContextIRLOrchestrator,
    ActorFromCIRLOrchestrator,
    get_batch, compute_tf_counts, estimate_loss, get_lr, configure_optimizers,
)
from lm_models import LMPolicyCritic, LMPolicyCriticConfig
from data.lm.prepare_rewards import prepare as prepare_rewards

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'lm')
)
VOCAB_SIZE     = 50304        # nanoGPT convention; GPT-2 BPE actual vocab is 50257
PROMPT         = "ROMEO:"
NUM_GEN_TOKENS = 80


def main():
    torch.manual_seed(0)
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    autocast_ctx = nullcontext()        # no AMP; testing transfer + train path correctness

    # Ensure tokenized data exists
    train_path = os.path.join(DATA_DIR, 'train.bin')
    val_path   = os.path.join(DATA_DIR, 'val.bin')
    assert os.path.exists(train_path), f"missing {train_path} - run data/lm/prepare.py first"
    assert os.path.exists(val_path),   f"missing {val_path} - run data/lm/prepare.py first"

    # Sizes shared by CIRL and actor (both consume the same Shakespeare data)
    V, N, B = VOCAB_SIZE, 64, 8
    K, D, gamma = 8, 10, 0.95

    # Reward-side checkpoint dir (Stage 1 writes here; Stage 1.5 reads from here) -- persistent
    cirl_ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ckpts', 'it_cirl'))
    os.makedirs(cirl_ckpt_dir, exist_ok=True)
    # Actor-side checkpoint dir (Stage 2 writes here; Stage 3 reads ckpt_best.pt from here)
    actor_ckpt_dir = tempfile.mkdtemp(prefix='it_actor_')
    # Reward .bin files land next to train.bin/val.bin (prepare_rewards enforces same-folder)
    train_rewards_path = os.path.join(DATA_DIR, 'train_rewards.bin')
    val_rewards_path   = os.path.join(DATA_DIR, 'val_rewards.bin')
    # CIRL ckpt path -- produced by Stage 1, consumed by Stage 1.5; final epoch with 5-epoch config
    cirl_ckpt_path = os.path.join(cirl_ckpt_dir, 'ckpt_epoch5.pt')
    # Skip stages 1 + 1.5 if IT_SKIP_CIRL=1 AND all prior artifacts exist
    SKIP_CIRL = (os.environ.get('IT_SKIP_CIRL', '0') == '1'
                 and os.path.exists(cirl_ckpt_path)
                 and os.path.exists(train_rewards_path)
                 and os.path.exists(val_rewards_path))

    print(f"data_dir={DATA_DIR}  cirl_ckpt_dir={cirl_ckpt_dir}  actor_ckpt_dir={actor_ckpt_dir}")
    if SKIP_CIRL:
        print(f"[IT_SKIP_CIRL=1] reusing CIRL ckpt at {cirl_ckpt_path} and existing rewards .bin files")

    if not SKIP_CIRL:
        # ============================================================
        # Stage 1 -- train a small RewardGPT via ContextIRLOrchestrator
        # ============================================================
        print("\n" + "=" * 12 + " STAGE 1: CIRL reward training " + "=" * 12)
        cirl_iters_per_epoch = 301_966 // (B * N)        # ~589
        cirl_max_iters = 10 * cirl_iters_per_epoch        # 5 epochs
        cirl_config = {
            'out_dir': cirl_ckpt_dir,
            'max_iters': cirl_max_iters,
            'iters_per_epoch': cirl_iters_per_epoch,
            'eval_interval': 100,
            'eval_iters': 10,
            'log_interval': 50,
            'eval_only': False,
            'always_save_checkpoint': False,
            'decay_lr': False,                            # constant LR throughout
            'learning_rate': 1e-3,
            'warmup_iters': 0,                            # unused when decay_lr=False
            'lr_decay_iters': cirl_max_iters,             # unused when decay_lr=False
            'min_lr': 1e-3,                               # harmonized with learning_rate
            'grad_clip': 1.0,
        }

        reward_conf = RewardGPTConfig(
            block_size=N, vocab_size=V, n_layer=2, n_head=4, n_embd=64,
        )                                                # is_causal=False (bidirectional)
        reward_net = RewardGPT(reward_conf).to(device)
        irl = ContextIRLReward(
            reward_net, V, K=K, D=D, gamma=gamma, surrogate_mode='uniform',
        ).to(device)
        cirl_optimizer = configure_optimizers(
            reward_net, weight_decay=0.1, learning_rate=cirl_config['learning_rate'],
            betas=(0.9, 0.95), device_type=device_type,
        )
        cirl_scaler = torch.cuda.amp.GradScaler(enabled=False)

        def cirl_get_batch(split):
            return get_batch(split, DATA_DIR, N, B, device, device_type)
        def cirl_estimate_loss():
            return estimate_loss(irl, cirl_get_batch, autocast_ctx, cirl_config['eval_iters'])
        def cirl_get_lr_fn(it):
            return get_lr(it, cirl_config['warmup_iters'], cirl_config['learning_rate'],
                          cirl_config['lr_decay_iters'], cirl_config['min_lr'])

        log_K = math.log(K)
        initial = cirl_estimate_loss()
        print(f"initial train loss {initial['train_loss']:.4f}  "
              f"(random-chance baseline log(K)={log_K:.3f})")

        ContextIRLOrchestrator(
            irl=irl, reward_net=reward_net, reward_conf=reward_conf,
            optimizer=cirl_optimizer, scaler=cirl_scaler, ctx=autocast_ctx,
            get_batch=cirl_get_batch,
            estimate_loss=cirl_estimate_loss, get_lr=cirl_get_lr_fn,
            config=cirl_config,
        ).start()

        final = cirl_estimate_loss()
        print(f"final train loss {final['train_loss']:.4f}  val loss {final['val_loss']:.4f}")
        assert final['train_loss'] < log_K * 0.5, (
            f"CIRL loss did not converge below 50% of log(K)={log_K:.3f}: "
            f"got {final['train_loss']:.3f}"
        )
        assert os.path.exists(cirl_ckpt_path), f"CIRL ckpt missing at {cirl_ckpt_path}"

        # ================================================================
        # Stage 1.5 -- precompute per-position rewards for train + val
        # ================================================================
        print("\n" + "=" * 12 + " STAGE 1.5: precompute rewards " + "=" * 12)
        prepare_rewards(train_path, train_rewards_path, cirl_ckpt_path)
        prepare_rewards(val_path,   val_rewards_path,   cirl_ckpt_path)

        # Verify the rewards files have the expected length (float32, same #positions as tokens)
        train_tokens_size = os.path.getsize(train_path)
        val_tokens_size   = os.path.getsize(val_path)
        assert os.path.getsize(train_rewards_path) == train_tokens_size * 2, \
            "train_rewards.bin size != train.bin size * 2 (uint16 -> float32)"
        assert os.path.getsize(val_rewards_path) == val_tokens_size * 2, \
            "val_rewards.bin size != val.bin size * 2 (uint16 -> float32)"

    # =====================================================
    # Stage 2 -- train LMPolicyCritic with cached rewards
    # =====================================================
    print("\n" + "=" * 12 + " STAGE 2: actor training " + "=" * 12)

    def actor_get_batch(split):
        """Sample a (B, N) window jointly from {split}.bin and {split}_rewards.bin."""
        tokens_path  = train_path  if split == 'train' else val_path
        rewards_path = train_rewards_path if split == 'train' else val_rewards_path
        token_data  = np.memmap(tokens_path,  dtype=np.uint16,  mode='r')
        reward_data = np.memmap(rewards_path, dtype=np.float32, mode='r')
        assert len(token_data) == len(reward_data), \
            f"token/reward length mismatch in {split}: {len(token_data)} != {len(reward_data)}"
        ix = torch.randint(len(token_data) - N, (B,))
        tokens  = torch.stack([torch.from_numpy((token_data [i:i+N]).astype(np.int64))  for i in ix])
        rewards = torch.stack([torch.from_numpy((reward_data[i:i+N]).copy())            for i in ix])
        if device_type == 'cuda':
            tokens  = tokens .pin_memory().to(device, non_blocking=True)
            rewards = rewards.pin_memory().to(device, non_blocking=True)
        else:
            tokens, rewards = tokens.to(device), rewards.to(device)
        return tokens, rewards

    actor_iters_per_epoch = 301_966 // (B * N)       # ~589 (same as CIRL)
    actor_max_iters = 5 * actor_iters_per_epoch      # 5 epochs
    actor_config = {
        'out_dir': actor_ckpt_dir,
        'max_iters': actor_max_iters,
        'iters_per_epoch': actor_iters_per_epoch,
        'eval_interval': 100,
        'eval_iters': 10,
        'log_interval': 50,
        'eval_only': False,
        'decay_lr': False,                            # constant LR throughout
        'learning_rate': 1e-4,
        'warmup_iters': 0,                            # unused when decay_lr=False
        'lr_decay_iters': actor_max_iters,            # unused when decay_lr=False
        'min_lr': 1e-3,                               # harmonized with learning_rate
        'grad_clip': 1.0,
        'gamma': gamma,                              # match Stage 1
        'use_target_net': True,                     # vanilla DQN
        'tau': 0.0005, #0.005
        'bootstrap_mode': 'argmax',
        'bootstrap_temperature': 1.0,
    }

    actor_model_cfg = LMPolicyCriticConfig(
        block_size=N, vocab_size=V, n_layer=2, n_head=4, n_embd=64,
    )                                                # is_causal=True by default
    policy_critic = LMPolicyCritic(actor_model_cfg).to(device)
    actor_optimizer = configure_optimizers(
        policy_critic, weight_decay=0.1, learning_rate=actor_config['learning_rate'],
        betas=(0.9, 0.95), device_type=device_type,
    )
    actor_scaler = torch.cuda.amp.GradScaler(enabled=False)

    def actor_get_lr_fn(it):
        return get_lr(it, actor_config['warmup_iters'], actor_config['learning_rate'],
                      actor_config['lr_decay_iters'], actor_config['min_lr'])

    orch = ActorFromCIRLOrchestrator(
        policy_critic=policy_critic,
        optimizer=actor_optimizer,
        scaler=actor_scaler,
        autocast_ctx=autocast_ctx,
        get_batch=actor_get_batch,
        get_lr=actor_get_lr_fn,
        config=actor_config,
    )

    initial_actor = orch._evaluate()
    print(f"initial val loss {initial_actor['val_loss']:.4f}  Q {initial_actor['val_Q']:.3f}")
    orch.start()
    final_actor = orch._evaluate()
    print(f"final   val loss {final_actor['val_loss']:.4f}  Q {final_actor['val_Q']:.3f}")
    assert final_actor['val_loss'] < initial_actor['val_loss'], (
        f"actor TD loss did not drop: initial {initial_actor['val_loss']:.4f} -> "
        f"final {final_actor['val_loss']:.4f}"
    )
    actor_best_ckpt = os.path.join(actor_ckpt_dir, 'ckpt_best.pt')
    assert os.path.exists(actor_best_ckpt), f"actor ckpt_best.pt missing at {actor_best_ckpt}"

    # =================================================================
    # Stage 3 -- load actor's best ckpt, generate, decode, print
    # =================================================================
    print("\n" + "=" * 12 + " STAGE 3: text generation " + "=" * 12)
    best = torch.load(actor_best_ckpt, map_location=device, weights_only=False)
    policy_critic.load_state_dict(best['policy_critic'])
    policy_critic.eval()

    enc = tiktoken.get_encoding('gpt2')
    prompt_ids = enc.encode_ordinary(PROMPT)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(NUM_GEN_TOKENS):
        idx_cond = idx if idx.size(1) <= actor_model_cfg.block_size \
                       else idx[:, -actor_model_cfg.block_size:]
        next_tok = orch.select_action(idx_cond, mode='sample', temperature=0.6, top_k=120)
        idx = torch.cat([idx, next_tok.unsqueeze(-1)], dim=1)

    print(f"\n--- generated continuation of {PROMPT!r} (first {NUM_GEN_TOKENS} tokens) ---")
    print(enc.decode(idx[0].tolist()))
    print(f"--- end ---\n")


if __name__ == '__main__':
    main()
