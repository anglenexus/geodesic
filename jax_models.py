import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from flax.training.train_state import TrainState


class Temperature(nn.Module):
    """Learnable temperature (entropy coefficient) for SAC."""
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full((1,), jnp.log(self.initial_temperature)),
        )

    def __call__(self):
        return jnp.exp(self.log_alpha)


class CriticTrainState(TrainState):
    """TrainState with target network parameters for soft updates."""
    target_params: dict


class ActorNetwork(nn.Module):
    """Policy network: outputs mean + log_std for each action dimension."""
    hidden_dim: int
    depth: int
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim * 2)(x)


class Ensemble(nn.Module):
    """Wraps a module class to create num independent copies via vmap.
    `net_kwargs` is a tuple of (name, value) pairs forwarded to net_cls — a tuple
    (not a dict) for Flax dataclass hashability."""
    net_cls: type
    num: int = 2
    net_kwargs: tuple = ()

    @nn.compact
    def __call__(self, *args):
        return nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,  # output: (batch, num, ...)
            axis_size=self.num,
        )(**dict(self.net_kwargs))(*args)


class JAXConcatQNetwork(nn.Module):
    """Q-network for a pre-assembled critic input, e.g. [obs | action | task_oh].
    The agent assembles the input upstream — same contract as MOORENetwork, so
    both plug in interchangeably as critic backbones."""
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)


# --- MOORE: Orthogonalized Mixture-Of-Experts (Hendawy et al., 2024) ---
# Ported from metaworld-algorithms/nn/moore.py. The MOORE network expects input
# of shape (B, obs_dim + num_tasks); the last num_tasks dims are a one-hot task ID.

def orthogonal_1d(x, num_experts):
    """Per-sample Gram-Schmidt across the expert axis. x: (B, num_experts, D)."""
    chex.assert_rank(x, 3)
    basis = jnp.expand_dims(
        x[:, 0, :] / (jnp.linalg.norm(x[:, 0, :], axis=1, keepdims=True) + 1e-8), axis=1
    )
    for i in range(1, num_experts):
        v = jnp.expand_dims(x[:, i, :], axis=1)
        w = v - ((v @ basis.transpose(0, 2, 1)) @ basis)
        wnorm = w / (jnp.linalg.norm(w, axis=2, keepdims=True) + 1e-8)
        basis = jnp.concatenate((basis, wnorm), axis=1)
    chex.assert_equal_shape((x, basis))
    return basis


class ExpertMLP(nn.Module):
    """One expert's torso: (depth-1) hidden ReLU layers + final linear projection.
    Equivalent to MLP(head_dim=width, depth=depth-1, width=width, activate_last=False)
    in the reference, with MOORE's defaults baked in (he_uniform, zeros bias, ReLU)."""
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.he_uniform()
        for _ in range(self.depth - 1):
            x = nn.relu(nn.Dense(self.width, kernel_init=kernel_init)(x))
        return nn.Dense(self.width, kernel_init=kernel_init)(x)


class MOORENetwork(nn.Module):
    """Multi-task MOORE network: orthogonalized mixture of experts + per-task head.
    Input  (B, obs_dim + num_tasks); the last num_tasks dims must be a one-hot task ID.
    Output (B, head_dim) — head_dim=1 for Q networks, 2*action_dim for policy.
    """
    num_tasks: int
    head_dim: int
    width: int = 400
    depth: int = 3
    num_experts: int = 6
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        batch_dim = x.shape[0]
        task_idx = x[..., -self.num_tasks:]
        x = x[..., : -self.num_tasks]

        # Task ID embedding — kernel_init = MOOREConfig.kernel_init = he_uniform.
        task_embedding = nn.Dense(
            self.num_experts,
            use_bias=False,
            kernel_init=jax.nn.initializers.he_uniform(),
        )(task_idx)

        # MOORE torso: num_experts parallel ExpertMLPs over the obs portion.
        experts_out = nn.vmap(
            ExpertMLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=-2,
            axis_size=self.num_experts,
        )(width=self.width, depth=self.depth)(x)

        experts_out = orthogonal_1d(experts_out, num_experts=self.num_experts)
        features_out = jnp.einsum("bnk,bn->bk", experts_out, task_embedding)
        features_out = jax.nn.tanh(features_out)

        # Per-task heads: num_tasks parallel Dense(head_dim).
        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=True,
        )(features_out)

        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]
        return x
