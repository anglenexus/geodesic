"""Unit tests for MOORE network + MooreSACAgent on MetaWorld MT10.

Network-level checks (cheap, no env):
  1. ut_moore_network_shape     — MOORENetwork forward output shape.
  2. ut_orthogonal_1d_orthonormal — orthogonal_1d returns orthonormal basis (Gram=I).
  3. ut_moore_mixture_parseval  — Parseval: ||sum w_i e_i||^2 == ||w||^2 for orthonormal e_i.
  4. ut_moore_per_task_routing  — different task one-hot selects different head output.

End-to-end check (mirrors ut_online_rl_metaworld_mt10_jax.py with MooreSACAgent):
  5. ut_mt10_online_jax_moore   — MT10 online RL training loop completes.
"""
import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")

import jax
import jax.numpy as jnp
import numpy as np

from jax_models import MOORENetwork, orthogonal_1d
from jax_agents import MooreSACAgent
from orchestrator import Orchestrator
from envs import MetaworldConfig


# ---------------------------------------------------------------------------
# Network-level tests
# ---------------------------------------------------------------------------

def ut_moore_network_shape():
    """MOORENetwork(B, obs_dim+num_tasks) -> (B, head_dim)."""
    num_tasks, obs_dim, head_dim, B = 10, 39, 8, 4
    net = MOORENetwork(num_tasks=num_tasks, head_dim=head_dim,
                       width=400, depth=3, num_experts=6)
    x = jnp.ones((B, obs_dim + num_tasks))
    params = net.init(jax.random.PRNGKey(0), x)
    out = net.apply(params, x)
    assert out.shape == (B, head_dim), f"expected ({B}, {head_dim}), got {out.shape}"
    print(f"  ut_moore_network_shape: OK (output {out.shape})")


def ut_orthogonal_1d_orthonormal():
    """orthogonal_1d output: rows along the expert axis are orthonormal per sample."""
    B, N, D = 8, 6, 32
    rng = np.random.default_rng(0)
    x = jnp.array(rng.standard_normal((B, N, D)).astype(np.float32))
    basis = orthogonal_1d(x, num_experts=N)

    # For each sample b: basis[b] @ basis[b].T should be approximately I_N.
    gram = jnp.einsum("bnd,bmd->bnm", basis, basis)
    eye = jnp.eye(N)[None, :, :]
    err = float(jnp.max(jnp.abs(gram - eye)))
    assert err < 1e-4, f"basis not orthonormal: max |gram - I| = {err}"
    print(f"  ut_orthogonal_1d_orthonormal: OK (max |gram - I| = {err:.2e})")


def ut_moore_mixture_parseval():
    """Parseval: mixing orthonormal experts with weights w preserves the norm.

    This directly tests the multi-expert sum step:
        features = einsum("bnk,bn->bk", experts_out, task_embedding)
    If experts are orthonormal across the expert axis, ||features||^2 == ||w||^2.
    A failure here means either orthogonalization or the einsum is wrong.
    """
    B, N, D = 4, 6, 32
    rng = np.random.default_rng(0)
    raw = jnp.array(rng.standard_normal((B, N, D)).astype(np.float32))
    basis = orthogonal_1d(raw, num_experts=N)              # (B, N, D), orthonormal
    w = jnp.array(rng.standard_normal((B, N)).astype(np.float32))
    mix = jnp.einsum("bnk,bn->bk", basis, w)               # (B, D)

    err = float(jnp.max(jnp.abs((mix ** 2).sum(-1) - (w ** 2).sum(-1))))
    assert err < 1e-3, f"Parseval violated: max |||mix||^2 - ||w||^2| = {err}"
    print(f"  ut_moore_mixture_parseval: OK (max Parseval err = {err:.2e})")


def ut_moore_per_task_routing():
    """Same input + same params, different task one-hot -> different output (different head selected)."""
    num_tasks, obs_dim, head_dim, B = 10, 39, 8, 1
    net = MOORENetwork(num_tasks=num_tasks, head_dim=head_dim,
                       width=64, depth=2, num_experts=4)
    obs = jnp.ones((B, obs_dim))
    oh0 = jnp.zeros((B, num_tasks)).at[:, 0].set(1.0)
    oh5 = jnp.zeros((B, num_tasks)).at[:, 5].set(1.0)
    x0 = jnp.concatenate([obs, oh0], axis=-1)
    x5 = jnp.concatenate([obs, oh5], axis=-1)

    params = net.init(jax.random.PRNGKey(0), x0)
    out0 = net.apply(params, x0)
    out5 = net.apply(params, x5)
    diff = float(jnp.max(jnp.abs(out0 - out5)))
    assert diff > 1e-3, f"task routing did not differentiate outputs (max diff {diff})"
    print(f"  ut_moore_per_task_routing: OK (max |out0 - out5| = {diff:.3f})")


# ---------------------------------------------------------------------------
# End-to-end test: MT10 online RL with MooreSACAgent
# ---------------------------------------------------------------------------

class MT10OnlineConfig:
    RL_MODE = "ONLINE"
    SEQ_LEN = 1
    STRIDE = 1
    BUFFER_CAPACITY = 10000
    TRAIN_BATCH_SIZE = 32
    SHUFFLE = False
    NUM_STEPS_PER_EPOCH = 50    # smoke-test-sized; baseline UT uses 500
    SEED = 42


def ut_mt10_online_jax_moore():
    """Smoke test: MT10 online RL loop with MooreSACAgent runs end-to-end."""
    env_config = MetaworldConfig(env_id="MT10", terminate_on_success=False)
    state_dim = env_config.observation_space.shape[0]   # 59 (49 obs + 10 one-hot)
    action_dim = int(np.prod(env_config.action_space.shape))  # 4
    agent = MooreSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=10,
        num_experts=6,
    )
    data_config = MT10OnlineConfig()
    orc = Orchestrator(
        agent=agent,
        data_config=data_config,
        num_epochs=2,             # smoke-test-sized; baseline UT uses 10
        env_config=env_config,
        warmstart_steps=50,
    )
    orc.start_online()
    print("  ut_mt10_online_jax_moore: OK")


if __name__ == "__main__":
    print("Running MOORE network unit tests...")
    ut_moore_network_shape()
    ut_orthogonal_1d_orthonormal()
    ut_moore_mixture_parseval()
    ut_moore_per_task_routing()
    print("All MOORE network unit tests passed.")
    print()
    print("Running MOORE MT10 end-to-end test...")
    ut_mt10_online_jax_moore()
    print("All tests passed.")
