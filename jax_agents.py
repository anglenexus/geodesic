import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import os
import flax.linen as nn
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from agents import SACAgent, Agent
from jax_models import JAXConcatQNetwork, ActorNetwork, Ensemble, Temperature, CriticTrainState, MOORENetwork


## [proposal] framework design protocal, jax accelerated code can be listed on the top section as pure functions 
## class structure is kept at the bottom section, retainin inheritance/abstractions including inheritance from pytorch friendly classes 

@jax.jit
def _select_actions(actor, states, key):
    """Pure function version of select_actions."""
    output = actor.apply_fn(actor.params, states)
    mean, log_std = jnp.split(output, 2, axis=-1)
    log_std = jnp.clip(log_std, -20, 2)
    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = distrax.Transformed(dist, distrax.Block(distrax.Tanh(), 1))
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    return actions, log_probs


# [to evaluate] consider taking out jax.vmap and use_task_vmap because there is no
# speedup and it's only useful when we split per task using task weights.
@jax.jit(static_argnames=['num_tasks', 'use_task_vmap'])
def _update_pure(actor, critic, alpha, key,
                 states, actions, rewards, next_states, dones,
                 num_tasks, use_task_vmap, gamma, tau, target_entropy):
    key, critic_key, actor_key = jax.random.split(key, 3)
    alpha_val = alpha.apply_fn(alpha.params)

    # Critic input format: [obs | action | task_oh]. Required by MOORE (which
    # routes via the trailing one-hot); a harmless rearrangement for plain MLP critics.
    def _q_in(s, a):
        return jnp.concatenate([s[..., :-num_tasks], a, s[..., -num_tasks:]], axis=-1)
    
    if use_task_vmap:
        # Reshape flat (N, feat) → (num_tasks, per_task_batch, feat)
        def reshape_by_task(x):
            return x.reshape(num_tasks, -1, *x.shape[1:])
        s_t, a_t, r_t, ns_t, d_t = map(reshape_by_task,
            [states, actions, rewards, next_states, dones])

    # --- 1. Critic update ---
    if use_task_vmap:
        # Compute target Q per task
        def compute_target(ns, r, d):
            next_actions, next_log_probs = _select_actions(actor, ns, critic_key)
            target_qs = critic.apply_fn(critic.target_params, _q_in(ns, next_actions))
            min_q = jnp.min(target_qs, axis=1).squeeze(-1)
            return r + gamma * (1 - d) * (min_q - alpha_val * next_log_probs)

        target_q = jax.lax.stop_gradient(
            jax.vmap(compute_target)(ns_t, r_t, d_t)
        )

        # Critic loss per task
        def critic_loss_fn(params, s, a, tq):
            q_pred = critic.apply_fn(params, _q_in(s, a))
            min_q_pred = jnp.min(q_pred, axis = 1).squeeze(-1)
            return ((min_q_pred - tq) ** 2).mean()

        critic_loss, critic_grads = jax.vmap(
            jax.value_and_grad(critic_loss_fn),
            in_axes=(None, 0, 0, 0),
        )(critic.params, s_t, a_t, target_q)
        critic_loss = critic_loss.mean()
        critic_grads = jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
    else:
        next_actions, next_log_probs = _select_actions(actor, next_states, critic_key)
        target_qs = critic.apply_fn(critic.target_params, _q_in(next_states, next_actions))
        min_q = jnp.min(target_qs, axis=1).squeeze(-1)
        target_q = rewards + gamma * (1 - dones) * (
            min_q - alpha_val * next_log_probs
        )
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss_fn(params):
            q_pred = critic.apply_fn(params, _q_in(states, actions))
            min_q_pred = jnp.min(q_pred, axis = 1).squeeze(-1)
            return ((min_q_pred - target_q) ** 2).mean()

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic.params)

    critic = critic.apply_gradients(grads=critic_grads)

    # --- 2. Actor update ---
    if use_task_vmap:
        def actor_loss_fn(actor_params, s):
            a, lp = _select_actions(actor.replace(params=actor_params), s, actor_key)
            q_vals = critic.apply_fn(critic.params, _q_in(s, a))
            min_q = jnp.min(q_vals, axis=1)
            return (alpha_val * lp - min_q).mean(), lp

        (actor_loss, log_probs), actor_grads = jax.vmap(
            jax.value_and_grad(actor_loss_fn, has_aux=True),
            in_axes=(None, 0),
        )(actor.params, s_t)
        actor_loss = actor_loss.mean()
        log_probs = log_probs.reshape(-1)
        actor_grads = jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
    else:
        def actor_loss_fn(actor_params):
            a, lp = _select_actions(actor.replace(params=actor_params), states, actor_key)
            q_vals = critic.apply_fn(critic.params, _q_in(states, a))
            min_q = jnp.min(q_vals, axis=1)
            return (alpha_val * lp - min_q).mean(), lp

        (actor_loss, log_probs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor.params)

    actor = actor.apply_gradients(grads=actor_grads)

    # --- 3. Alpha (temperature) update ---
    def alpha_loss_fn(alpha_params):
        log_alpha = alpha_params["params"]["log_alpha"]
        return (-log_alpha * (log_probs + target_entropy)).mean()

    alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha.params)
    alpha = alpha.apply_gradients(grads=alpha_grads)

    # --- 4. Soft update targets ---
    critic = critic.replace(
        target_params=optax.incremental_update(
            critic.params, critic.target_params, tau
        )
    )

    logs = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'sac_alpha': alpha_val.squeeze(),
    }
    return actor, critic, alpha, key, logs


class JAXSACAgent(SACAgent):
    """
    JAX implementation of SAC, inheriting from SACAgent (PyTorch).
    Overrides all framework-specific methods while reusing:
    _transform_actions, compute_cql_loss, get_save_path, train.
    """

    def __init__(self, state_dim, action_dim, num_tasks=10, use_task_vmap=False,
                 hidden_dim=400, depth=3, lr=3e-3, gamma=0.95, tau=0.005, seed=42,
                 actor_lr=None, critic_lr=None, alpha_lr=None, max_grad_norm=None):
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.use_task_vmap = use_task_vmap
        self.hidden_dim = hidden_dim
        self.depth = depth
        Agent.__init__(self)  # skip torch init in BaseQLAgent/SACAgent
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.training_step = 0
        self.target_entropy = -action_dim
        # Per-component learning rates fall back to shared `lr` if not specified.
        actor_lr = actor_lr if actor_lr is not None else lr
        critic_lr = critic_lr if critic_lr is not None else lr
        alpha_lr = alpha_lr if alpha_lr is not None else lr

        # Optimizer factory: optionally clip global grad norm (actor/critic only;
        # the reference does not clip the temperature optimizer).
        def _opt(lr_val, clip=False):
            adam = optax.adam(lr_val)
            if clip and max_grad_norm is not None:
                return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)
            return adam

        # RNG key management (replaces PyTorch global RNG)
        key = jax.random.PRNGKey(seed)
        key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)
        self.key = key

        # Actor: self.actor + self.actor_optimizer → TrainState
        actor_net = self._make_actor(state_dim, action_dim)
        dummy_obs = jnp.ones((1, state_dim))
        self.actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_obs),
            tx=_opt(actor_lr, clip=True),
        )

        # Temperature: self.log_alpha + self.alpha_optimizer → TrainState
        alpha_net = Temperature()
        self.alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_key),
            tx=_opt(alpha_lr, clip=False),
        )

        # Critics: self.q1/q2 + targets + optimizers → CriticTrainState
        # Critic input is the assembled [obs | action | task_oh] (see _q_in in _update_pure).
        critic_net = self._make_critic(state_dim)
        dummy_q_in = jnp.ones((1, state_dim + action_dim))
        critic_params = critic_net.init(critic_key, dummy_q_in)
        self.critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            target_params=critic_params,
            tx=_opt(critic_lr, clip=True),
        )

    def _make_actor(self, state_dim, action_dim):
        return ActorNetwork(self.hidden_dim, self.depth, action_dim)

    def _make_critic(self, state_dim):
        return Ensemble(
            net_cls=JAXConcatQNetwork,
            num=2,
            net_kwargs=(("hidden_dim", self.hidden_dim), ("depth", self.depth)),
        )

    def sample_action(self, obs):
        self.key, action_key = jax.random.split(self.key)
        actions, _ = _select_actions(self.actor, obs, action_key)
        return np.asarray(actions)

    def update(self, states, actions, rewards, next_states, dones):
        # Convert from torch/numpy to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)

        # Call JIT-compiled pure function
        self.actor, self.critic, self.alpha, self.key, logs = _update_pure(
            self.actor, self.critic, self.alpha, self.key,
            states, actions, rewards, next_states, dones,
            self.num_tasks, self.use_task_vmap, self.gamma, self.tau, self.target_entropy,
        )

        self.training_step += 1

        return {k: float(v) for k, v in logs.items()}

    def save(self, path, epoch=None, data_config=None):
        path = os.path.abspath(path)
        data_config_name = type(data_config).__name__ if data_config else ''
        ckpt = {
            'actor': self.actor,
            'critic': self.critic,
            'alpha': self.alpha,
            'key': self.key,
            'training_step': np.array(self.training_step),
            'epoch': np.array(epoch or 0),
            'data_config': np.array(list(data_config_name.encode('utf-8')), dtype=np.uint8),
        }
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, ckpt)
        checkpointer.wait_until_finished()
        checkpointer.close() ## uses async io, ensure checkpointer is closed or error 
        print(f"Model saved to {path}")

    def load(self, path):
        path = os.path.abspath(path)
        target = {
            'actor': self.actor,
            'critic': self.critic,
            'alpha': self.alpha,
            'key': self.key,
            'training_step': np.array(self.training_step),
            'epoch': np.array(0),
            'data_config': np.array([], dtype=np.uint8),
        }
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(path, target)
        self.actor = restored['actor']
        self.critic = restored['critic']
        self.alpha = restored['alpha']
        self.key = restored['key']
        self.training_step = int(restored['training_step'])
        epoch = int(restored['epoch'])
        data_config_name = bytes(restored['data_config'].tolist()).decode('utf-8')
        print(f"Model loaded from {path}")
        return epoch, data_config_name


class MooreSACAgent(JAXSACAgent):
    """SAC with the MOORE (Orthogonalized Mixture-Of-Experts) network for both
    actor and critic. Inherits the SAC training loop from JAXSACAgent unchanged;
    only the network factories change.

    The actor takes states directly (states already include the trailing task
    one-hot). The critic receives the assembled [obs | action | task_oh] from
    _update_pure._q_in, and routes via the trailing one-hot just like the actor.
    """

    def __init__(self, *args, num_experts: int = 6, **kwargs):
        # Set num_experts before super().__init__ so _make_actor / _make_critic see it.
        self.num_experts = num_experts
        super().__init__(*args, **kwargs)

    def _make_actor(self, state_dim, action_dim):
        return MOORENetwork(
            num_tasks=self.num_tasks,
            head_dim=action_dim * 2,   # mean + log_std for each action dim
            width=self.hidden_dim,
            depth=self.depth,
            num_experts=self.num_experts,
        )

    def _make_critic(self, state_dim):
        return Ensemble(
            net_cls=MOORENetwork,
            num=2,
            net_kwargs=(
                ("num_tasks", self.num_tasks),
                ("head_dim", 1),                  # Q value
                ("width", self.hidden_dim),
                ("depth", self.depth),
                ("num_experts", self.num_experts),
            ),
        )
