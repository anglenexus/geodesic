"""Transformer building blocks and the policy/critic LM (lm_models.py).

Houses the shared nanoGPT-style components (LayerNorm, MLP, SelfAttention,
Block) used by both cirl_lm_orchestrator.RewardGPT (bidirectional) and
LMPolicyCritic (causal). SelfAttention is parameterized by config.is_causal
so the same class serves both call sites.

LMPolicyCritic is a DQN-style policy/critic with a vocab-sized output head:
    mode='train' -> ReLU(linear(x))   — per-token Q values (>= 0).
    mode='infer' -> Softmax(linear(x)) — next-token policy distribution.

Model code adapted from nanoGPT (https://github.com/karpathy/nanoGPT, MIT).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias.  (nanoGPT) """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """ nanoGPT's CausalSelfAttention, parameterized by config.is_causal. """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout
        self.is_causal = config.is_causal
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            if self.is_causal:
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):                                  # nanoGPT
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):                                # nanoGPT
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class LMPolicyCriticConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer:    int = 4
    n_head:     int = 4
    n_embd:     int = 128
    dropout:  float = 0.0
    bias:      bool = False
    is_causal:  bool = True


class LMPolicyCritic(nn.Module):
    """nanoGPT-style causal transformer with a vocab-sized Q head.

    Contract:
        forward(idx: LongTensor (B, N), mode='train') -> Tensor (B, N, V)
            mode='train': returns ReLU(q_head(x)) — per-token Q values >= 0.
            mode='infer': returns Softmax(q_head(x)) — policy distribution.
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
        # No weight tying between wte and q_head: Q values and token-embedding
        # similarities are semantically distinct objectives.
        self.q_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"lm policy/critic parameters: {self.get_num_params()/1e6:.3f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, mode='train'):
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"sequence length {t} > block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if mode == 'train':
            return 5 * F.sigmoid(self.q_head(x)) #F.relu(self.q_head(x))        # (B, N, V) Q values
        elif mode == 'infer':
            return F.softmax(self.q_head(x[:, [-1], :]), dim=-1)       # (B, 1, V) policy on last pos
        else:
            raise ValueError(mode)


# TODO: action-input critic variant (classical Q-learning).
# Mirror jax_models.JAXConcatQNetwork: instead of returning Q for all V actions
# at once, take (state_tokens, action_token) and return a single scalar Q. Train
# against action-space surrogates the same way ContextIRLReward samples K-1
# negatives. Useful when V is too large for a vocab-sized output head, or when
# we want the SAC actor/critic decomposition rather than the DQN argmax policy.
