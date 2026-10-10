"""Context-IRL library: reward model + IRL wrapper + training orchestrator.

Classes:
    RewardGPT / RewardGPTConfig — nanoGPT-style transformer with bidirectional
        attention and a per-token reward head (Linear -> sigmoid). Maps
        (B, N) token ids to (B, N) per-position rewards in (0, 1).
    ContextIRLReward — Context-IRL loss wrapper. Given a reward_net, samples
        K candidates per position (true + K-1 surrogates from tf or in_seq)
        and computes cross-entropy over discounted future rewards.
    ContextIRLOrchestrator — nanoGPT-style training loop. Takes the IRL
        wrapper, optimizer, scaler, ctx, data callables, and a config dict;
        run via .start().

Free functions (callable with explicit args, no module-level globals):
    get_batch(split, data_dir, block_size, batch_size, device, device_type)
    compute_tf_counts(data_dir, vocab_size)
    estimate_loss(irl, get_batch_fn, ctx, eval_iters)
    get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr)

Model code adapted from nanoGPT (https://github.com/karpathy/nanoGPT, MIT):
sites where Context-IRL diverges from upstream are marked with ``### MOD:``.
"""

import os
import math
import time
import copy
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_models import LayerNorm, SelfAttention, MLP, Block


@dataclass
class RewardGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64
    dropout:  float = 0.0
    bias:      bool = False
    is_causal:  bool = False                # preserves bidirectional attention (Context-IRL default)


class RewardGPT(nn.Module):
    """
    nanoGPT's GPT, modified at two sites:
      ### MOD 1: SelfAttention is bidirectional (see above)
      ### MOD 2: lm_head -> reward_head: Linear(n_embd, 1) + sigmoid, no weight tying.
    Contract: forward(idx: LongTensor (B, N)) -> Tensor (B, N) in (0, 1).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.reward_head = nn.Linear(config.n_embd, 1, bias=config.bias)   ### MOD 2

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"reward model parameters: {self.get_num_params()/1e6:.3f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return torch.sigmoid(self.reward_head(x)).squeeze(-1)              ### MOD 2


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """nanoGPT-style AdamW: 2D weights get weight-decayed, <2D don't; fused on CUDA."""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params   = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() <  2]
    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    num_decay   = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer


# =============================================================================
# CONTEXT-IRL WRAPPER
# =============================================================================

class ContextIRLReward(nn.Module):
    """
    Context-IRL reward learning over token sequences (LM-style, stateless).

    Wraps a transformer that maps (B, N) token ids -> (B, N) per-position rewards.
    """
    def __init__(self, reward_net, vocab_size, K=8, D=10, gamma=0.99,
                 surrogate_mode='uniform', tf_counts: torch.Tensor | None = None):
        super().__init__()
        self.reward_net = reward_net
        self.vocab_size = vocab_size
        self.K, self.D, self.gamma = K, D, gamma
        self.surrogate_mode = surrogate_mode
        if surrogate_mode == 'tf':
            assert tf_counts is not None and tf_counts.numel() == vocab_size
            probs = tf_counts.float() / tf_counts.float().sum()
            self.register_buffer('tf_probs', probs)

    def _surrogates_uniform(self, B, num, device):
        return torch.randint(0, self.vocab_size, (B, num), device=device)

    def _surrogates_tf(self, B, num, device):
        idx = torch.multinomial(self.tf_probs, B * num, replacement=True)
        return idx.view(B, num).to(device)

    def _surrogates_in_seq(self, tokens, pos, num):
        B, N = tokens.shape
        r = torch.randint(0, N - 1, (B, num), device=tokens.device)
        r = r + (r >= pos.unsqueeze(1)).long()
        return tokens.gather(1, r)

    def forward(self, tokens):
        """
        tokens: (B, N) LongTensor
        returns: (loss, acc) where acc is top-1 accuracy on the K-way ranking
        """
        B, N  = tokens.shape
        K, D, g = self.K, self.D, self.gamma
        device = tokens.device

        pos      = torch.randint(0, N - D + 1, (B,), device=device)            # (B,)
        true_tok = tokens.gather(1, pos.unsqueeze(1)).squeeze(1)               # (B,)
        if self.surrogate_mode == 'uniform':
            surr = self._surrogates_uniform(B, K - 1, device)
        elif self.surrogate_mode == 'tf':
            surr = self._surrogates_tf(B, K - 1, device)
        elif self.surrogate_mode == 'in_seq':
            surr = self._surrogates_in_seq(tokens, pos, K - 1)
        else:
            raise ValueError(self.surrogate_mode)
        cands = torch.cat([true_tok.unsqueeze(1), surr], dim=1)                # (B, K)

        expanded = tokens.unsqueeze(1).expand(B, K, N).clone()                 # (B, K, N)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        k_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        p_idx = pos.unsqueeze(1).expand(B, K)
        expanded[b_idx, k_idx, p_idx] = cands

        r_pred = self.reward_net(expanded.view(B * K, N)).view(B, K, N)        # (B, K, N)

        j      = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        offset = j - pos.unsqueeze(1)
        valid  = (offset >= 0).to(r_pred.dtype)
        disc   = (g ** offset.clamp(min=0).to(r_pred.dtype)) * valid           # (B, N)
        Q      = (r_pred * disc.unsqueeze(1)).sum(dim=-1)                      # (B, K)

        target = torch.zeros(B, dtype=torch.long, device=device)
        loss   = F.cross_entropy(Q, target)
        with torch.no_grad():
            acc = (Q.argmax(dim=1) == target).float().mean()
        return loss, acc


# =============================================================================
# ORCHESTRATOR  (training loop, nanoGPT-style)
# =============================================================================

class ContextIRLOrchestrator:
    """Trains a Context-IRL reward model with a nanoGPT-style loop."""
    def __init__(
        self,
        irl,
        reward_net,
        reward_conf,
        optimizer,
        scaler,
        ctx,
        get_batch,
        estimate_loss,
        get_lr,
        config,
    ):
        self.irl = irl
        self.reward_net = reward_net
        self.reward_conf = reward_conf
        self.optimizer = optimizer
        self.scaler = scaler
        self.ctx = ctx
        self.get_batch = get_batch
        self.estimate_loss = estimate_loss
        self.get_lr = get_lr
        self.config = config

    def start(self):
        c = self.config
        iter_num = 0
        best_val_loss = 1e9
        X, _ = self.get_batch('train')
        t0 = time.time()
        print('-' * 10 + 'starting training' + '-' * 10)

        while True:
            lr = self.get_lr(iter_num) if c['decay_lr'] else c['learning_rate']
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            if iter_num % c['eval_interval'] == 0:
                m = self.estimate_loss()
                print(f"step {iter_num:>6d} | "
                      f"train loss {m['train_loss']:.4f} acc {m['train_acc']:.3f}  |  "
                      f"val loss {m['val_loss']:.4f} acc {m['val_acc']:.3f}  |  lr {lr:.2e}")
                if (m['val_loss'] < best_val_loss or c['always_save_checkpoint']) and iter_num > 0:
                    best_val_loss = min(best_val_loss, m['val_loss'])
                    ckpt = {
                        'reward_net':    self.reward_net.state_dict(),
                        'optimizer':     self.optimizer.state_dict(),
                        'reward_conf':   self.reward_conf.__dict__,
                        'iter_num':      iter_num,
                        'best_val_loss': best_val_loss,
                        'config':        c,
                    }
                    iters_per_epoch = c.get('iters_per_epoch')
                    fname = (f'ckpt_epoch{iter_num // iters_per_epoch + 1}.pt'
                             if iters_per_epoch else 'ckpt.pt')
                    torch.save(ckpt, os.path.join(c['out_dir'], fname))
                    print(f"  saved checkpoint -> {c['out_dir']}/{fname}")

            if iter_num == 0 and c['eval_only']:
                break

            with self.ctx:
                loss, acc = self.irl(X)
            X, _ = self.get_batch('train')           # async prefetch (nanoGPT pattern)
            self.scaler.scale(loss).backward()
            if c['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), c['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time(); dt = t1 - t0; t0 = t1
            if iter_num % c['log_interval'] == 0:
                print(f"iter {iter_num:>6d}: loss {loss.item():.4f}  acc {acc.item():.3f}  "
                      f"lr {lr:.2e}  dt {dt*1000:.0f}ms")
            iter_num += 1
            if iter_num > c['max_iters']:
                break


class ActorFromCIRLOrchestrator:
    """Train an LMPolicyCritic actor against pre-computed per-position rewards.

    DQN-style TD loss (per-batch, vectorized over time t = 1..N-1):
        target_t = r(t+1) + gamma * max_{a'} Q_{tgt}(s_{t+1}, a')        # detached
        loss     = ((Q_t - target_t) ** 2).mean()
    where Q_t = output[:, t-1, tokens[:, t]] from the policy/critic forward.

    Rewards must arrive pre-computed via get_batch(split) -> (tokens, rewards),
    each (B, N). The reward model itself never appears here — it ran offline
    (see data/lm/prepare_rewards.py for the pre-caching pipeline).

    Bootstrap source for max_{a'} Q_{tgt}:
        config['use_target_net']=False (default)  -> online policy_critic (detached).
        config['use_target_net']=True             -> Polyak-averaged target net at
                                                     rate config['tau'] per step.
    """
    def __init__(
        self,
        policy_critic,            # LMPolicyCritic — combined actor + critic
        optimizer,
        scaler,
        autocast_ctx,             # autocast manager (nullcontext on CPU)
        get_batch,                # callable: split -> (tokens, rewards) both (B, N)
        get_lr,
        config,
    ):
        if getattr(getattr(policy_critic, 'config', None), 'is_causal', None) is not True:
            raise ValueError(
                "ActorFromCIRLOrchestrator requires a causal policy_critic; "
                "set is_causal=True on its config (e.g. LMPolicyCriticConfig)."
            )
        self.policy_critic = policy_critic
        self.optimizer = optimizer
        self.scaler = scaler
        self.autocast_ctx = autocast_ctx
        self.get_batch = get_batch
        self.get_lr = get_lr
        self.config = config
        self.gamma                 = config['gamma']
        self.use_target_net        = config.get('use_target_net', False)
        self.tau                   = config.get('tau', 0.005)
        self.bootstrap_mode        = config.get('bootstrap_mode', 'argmax')
        self.bootstrap_temperature = config.get('bootstrap_temperature', 1.0)
        self.target_net = None
        if self.use_target_net:
            self.target_net = copy.deepcopy(policy_critic).eval()
            for p in self.target_net.parameters():
                p.requires_grad_(False)

    def _td_loss(self, tokens, rewards):
        """TD loss for one (tokens, rewards) batch.

        Bootstrap modes (config['bootstrap_mode']):
            'argmax'   (default) — Q_next = max_{a'} Q(s', a').
            'expected'           — Q_next = sum_{a'} softmax(Q(s', .) / T)_{a'} * Q(s', a').
        """
        Q = self.policy_critic(tokens, mode='train')                              # (B, N, V), ReLU'd
        predicted_Q = Q[:, :-1, :].gather(2, tokens[:, 1:].unsqueeze(2)).squeeze(2)  # (B, N-1)
        with torch.no_grad():
            src = self.target_net if self.target_net is not None else self.policy_critic
            Q_next_all = src(tokens, mode='train')[:, 1:, :]                       # (B, N-1, V)
            if self.bootstrap_mode == 'argmax':
                Q_next = Q_next_all.max(dim=-1).values                              # (B, N-1)
            elif self.bootstrap_mode == 'expected':
                probs  = torch.softmax(Q_next_all / self.bootstrap_temperature, dim=-1)
                Q_next = (probs * Q_next_all).sum(dim=-1)                           # (B, N-1)
            else:
                raise ValueError(self.bootstrap_mode)
        target = rewards[:, 1:] + self.gamma * Q_next
        loss = ((predicted_Q - target.detach()) ** 2).mean()
        return loss, predicted_Q.detach().mean()       # (loss, Q): Q is mean predicted Q over the batch

    def select_action(self, tokens, mode='argmax', temperature=1.0, top_k=None):
        """Pick next action given a context sequence (text-generation interface).

        tokens:      (B, T) LongTensor — current context, T <= block_size.
        mode:        'argmax' (deterministic) or 'sample' (stochastic).
        temperature: scales Q before softmax in 'sample' mode.
        top_k:       if int, restrict 'sample' to top-k Q values (nanoGPT-style).
        returns:     (B,) LongTensor of selected next tokens.
        """
        with torch.no_grad():
            Q_last = self.policy_critic(tokens, mode='train')[:, -1, :]            # (B, V), ReLU'd
            if mode == 'argmax':
                return Q_last.argmax(dim=-1)
            if mode == 'sample':
                logits = Q_last / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(-1)
            raise ValueError(mode)

    def _soft_update_target(self):
        """Polyak-averaged update of target_net params from policy_critic params.

        No-op when self.target_net is None (vanilla mode, no target network).
        """
        if self.target_net is None:
            return
        with torch.no_grad():
            for p_target, p_online in zip(self.target_net.parameters(),
                                          self.policy_critic.parameters()):
                # tensor.lerp_(other, weight) is in-place:
                #   self = self * (1 - weight) + other * weight
                # so p_target = (1 - tau) * p_target + tau * p_online — Polyak averaging.
                p_target.data.lerp_(p_online.data, self.tau)

    @torch.no_grad()
    def _evaluate(self):
        """Average TD loss + mean predicted Q over `eval_iters` val batches only.

        Returns dict with 'val_loss' and 'val_Q'. Higher val_Q means the
        policy/critic predicts higher expected return on held-out data.
        Train metrics are reported from the live training step, not re-computed here.
        """
        c = self.config
        self.policy_critic.eval()
        losses, Qs = torch.zeros(c['eval_iters']), torch.zeros(c['eval_iters'])
        for i in range(c['eval_iters']):
            tokens, rewards = self.get_batch('val')
            with self.autocast_ctx:
                loss, Q = self._td_loss(tokens, rewards)
            losses[i], Qs[i] = loss.item(), Q.item()
        self.policy_critic.train()
        return {'val_loss': losses.mean().item(), 'val_Q': Qs.mean().item()}

    def start(self):
        c = self.config
        iter_num = 0
        best_val_loss = float('inf')                  # global minimum across all training
        iters_per_epoch = c.get('iters_per_epoch')    # optional; only affects progress filename
        t0 = time.time()
        print('-' * 10 + 'starting actor training' + '-' * 10)

        while True:
            # --- LR schedule for this iter ---
            lr = self.get_lr(iter_num) if c['decay_lr'] else c['learning_rate']
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # --- val eval + dual checkpointing at every eval_interval ---
            if iter_num % c['eval_interval'] == 0:
                m = self._evaluate()
                print(f"step {iter_num:>6d} | "
                      f"val loss {m['val_loss']:.4f} Q {m['val_Q']:.3f}  |  lr {lr:.2e}")
                if iter_num > 0:
                    ckpt = {
                        'policy_critic': self.policy_critic.state_dict(),
                        'optimizer':     self.optimizer.state_dict(),
                        'iter_num':      iter_num,
                        'val_loss':      m['val_loss'],
                        'val_Q':         m['val_Q'],
                        'config':        c,
                    }
                    # (1) progress snapshot — overwritten within each epoch
                    if iters_per_epoch:
                        progress_fname = f'ckpt_epoch{iter_num // iters_per_epoch + 1}.pt'
                    else:
                        progress_fname = 'ckpt_progress.pt'
                    torch.save(ckpt, os.path.join(c['out_dir'], progress_fname))
                    print(f"  saved progress -> {c['out_dir']}/{progress_fname}")
                    # (2) global best — single file, overwritten only when val_loss is below all-time min
                    if m['val_loss'] < best_val_loss:
                        best_val_loss = m['val_loss']
                        ckpt['best_val_loss'] = best_val_loss
                        torch.save(ckpt, os.path.join(c['out_dir'], 'ckpt_best.pt'))
                        print(f"  saved best     -> {c['out_dir']}/ckpt_best.pt (val_loss={best_val_loss:.4f})")

            if iter_num == 0 and c['eval_only']:
                break

            # --- training step (linear order: fetch batch -> compute loss -> step) ---
            X, R = self.get_batch('train')
            with self.autocast_ctx:
                loss, Q = self._td_loss(X, R)
            self.scaler.scale(loss).backward()
            if c['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_critic.parameters(), c['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self._soft_update_target()                # no-op when target_net is None

            # --- per-iter logging (train side) ---
            t1 = time.time(); dt = t1 - t0; t0 = t1
            if iter_num % c['log_interval'] == 0:
                print(f"iter {iter_num:>6d}: train loss {loss.item():.4f}  Q {Q.item():.3f}  "
                      f"lr {lr:.2e}  dt {dt*1000:.0f}ms")
            iter_num += 1
            if iter_num > c['max_iters']:
                break


# =============================================================================
# DATA / EVAL / LR HELPERS (free functions; explicit args, no globals)
# =============================================================================

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    """nanoGPT-style memmap batch sampler. Returns (x, y) of shape (B, N)."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))     for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def compute_tf_counts(data_dir, vocab_size):
    """Stream train.bin through np.bincount in chunks (works for OpenWebText too)."""
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    counts = np.zeros(vocab_size, dtype=np.int64)
    chunk = 10_000_000
    for i in range(0, len(data), chunk):
        counts += np.bincount(np.asarray(data[i:i+chunk], dtype=np.int64), minlength=vocab_size)
    counts = counts.astype(np.float32) + 1.0          # Laplace smoothing
    print(f"computed unigram counts over {len(data):,} tokens; vocab seen: {(counts > 1).sum()}")
    return torch.from_numpy(counts)


@torch.no_grad()
def estimate_loss(irl, get_batch_fn, ctx, eval_iters):
    """Average (loss, acc) over `eval_iters` batches on train and val splits."""
    out = {}
    irl.eval()
    for split in ('train', 'val'):
        losses, accs = torch.zeros(eval_iters), torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, _ = get_batch_fn(split)
            with ctx:
                loss, acc = irl(x)
            losses[i], accs[i] = loss.item(), acc.item()
        out[f'{split}_loss'] = losses.mean().item()
        out[f'{split}_acc']  = accs.mean().item()
    irl.train()
    return out


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Linear warmup + cosine decay schedule from nanoGPT."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (learning_rate - min_lr)
