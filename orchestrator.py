import torch, time, numpy as np, ray
import jax, jax.numpy as jnp, optax
from jax_models import ActorNetwork
from jax_agents import _select_actions
from flax.training.train_state import TrainState
from datasets import OfflineRLDataCollection, SequenceDataCollection, ParallelReplayBuffer
from models import ModelSharedStorage

class Orchestrator:
    def __init__(
        self,
        agent,
        data_config,
        num_epochs=None,
        env_config=None,
        warmstart_steps=None,
    ):
        self.agent = agent
        self.data_config = data_config
        self.num_epochs = num_epochs
        self.warmstart_steps = warmstart_steps
        self.num_steps_per_epoch = getattr(data_config, 'NUM_STEPS_PER_EPOCH', None)

        if data_config.RL_MODE == "ONLINE":
            self.data_collection = SequenceDataCollection(data_config)
            self.envs = env_config.spawn()
        else:
            self.data_collection = OfflineRLDataCollection(data_config)
    
    def start(self): 
        # ------------------------------------------------------------------
        # start()
        # ------------------------------------------------------------------
        # start orchestration, includes environment play and training. 
        # Shared structure:
        #   1. Print header / hyperparameters
        start_time = time.time()
        print('-'*10 + 'starting training' + '-'*10)
        for epoch in range(self.num_epochs): 
            for batch_idx, batch in enumerate(self.data_collection.train_loader):
                if batch_idx % 100 == 0: 
                    print(str(batch_idx), end='..') 
                #print(features)
                train_metrics = self.agent.update(
                    batch[0], # states 
                    batch[1], # actions
                    batch[2], # rewards 
                    batch[3], # next_states 
                    batch[4], # dones 
                )
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += train_metrics.get(key, 0)
            
            # Average metrics
            for key in train_metrics:
                train_metrics[key] /= len(self.data_collection.train_loader)
            
            print('\n'+'-'*5 + f'validating ... training done for epoch:{epoch}' + '-'*5)
            val_batch = next(iter(self.data_collection.val_loader))
            val_q = self.agent.validate(
                val_batch[0],
                val_batch[1],
            )
            print('<'*2 + f'validation q: {val_q}' + '<'*2)
            # Log progress 
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_td_loss = (train_metrics['q1_loss'] + train_metrics['q2_loss']) / 2
                print(f"Epoch {epoch+1}: "
                    f"TD Loss={avg_td_loss:.4f} (Q1={train_metrics['q1_loss']:.4f}, Q2={train_metrics['q2_loss']:.4f}), "
                    #f"CQL Loss (Q1={train_metrics['cql1_loss']:.4f}, Q2={train_metrics['cql2_loss']:.4f}), "
                    f"Val Q={val_q:.4f}, "
                    f"Best Val Q={self.agent.best_val_q:.4f}, "
                    f"Time={elapsed/60:.1f}min", flush=True)
            
        # Save final model
        self.agent.save(self.agent.get_save_path('final'))
        
        total_time = time.time() - start_time
        print(f"\n({self.agent.experiment_prefix}) completed in {total_time/60:.1f} minutes!", flush=True)
        print("Models saved:", flush=True)
        print(f"  - {self.agent.get_save_path('best')}", flush=True)
        print(f"  - {self.agent.get_save_path('final')}", flush=True)

    def start_online(self):
        obs, _ = self.envs.reset()
        start_time = time.time()
        print('-'*10 + 'starting online RL training' + '-'*10)

        global_step = 0
        for epoch in range(self.num_epochs):
            for step in range(self.num_steps_per_epoch):
                if step % 200 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"  ep{epoch+1} step {step}/{self.num_steps_per_epoch} "
                          f"(global={global_step}, time={elapsed:.1f}min)", flush=True)
                if global_step < self.warmstart_steps:
                    actions = self.envs.action_space.sample()
                else:
                    actions = self.agent.sample_action(obs)

                # envs.step expects numpy arrays on CPU
                if torch.is_tensor(actions):
                    actions = actions.cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                dones = np.logical_or(terminations, truncations)

                # Use true terminal obs for buffer, not the auto-reset obs
                buffer_obs = next_obs
                if "final_obs" in infos:
                    buffer_obs = np.where(
                        dones[:, None], np.stack(infos["final_obs"]), next_obs
                    )

                self.data_collection.dataset.add_transition(obs, actions, rewards, buffer_obs, dones)
                obs = next_obs

                # Create train_loader once we have enough samples
                if self.data_collection.data_loader is None \
                        and len(self.data_collection.dataset) >= self.data_collection.batch_size:
                    self.data_collection.create_train_loader()

                if global_step >= self.warmstart_steps and self.data_collection.data_loader is not None:
                    batch = self.data_collection.sample_batch()
                    # Flatten (batch, seq_len, num_tasks, feat_dim) → (N, feat_dim)
                    # e.g. states: (32,1,10,59)→(320,59), rewards: (32,1,10)→(320,)
                    batch = tuple(
                        b.reshape(-1, *b.shape[3:]) if b.dim() > 2 else b.reshape(-1)
                        for b in batch
                    )
                    self.agent.update(*batch)

                global_step += 1

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"global_step={global_step}, "
                  f"sequences={len(self.data_collection.dataset)}, "
                  f"time={elapsed/60:.1f}min")

        self.envs.close()
        print(f"Online training complete in {(time.time()-start_time)/60:.1f}min")


@ray.remote
class DataWorker:
    """Ray actor that collects transitions by interacting with the environment.

    Each worker owns its own env instance and a local copy of the actor
    network (params only, no optimizer). It pulls fresh actor params from
    ModelSharedStorage at a configurable frequency, and pushes transitions
    into the shared Ray Queue for the ParallelReplayBuffer to drain.
    """
    def __init__(self, rank, queue, model_storage, env_config,
                 state_dim, action_dim, hidden_dim=400, depth=3,
                 warmstart_steps=0, worker_weight_ckpt_interval=100,
                 queue_high_water=800, queue_low_water=200):
        self.rank = rank
        self.queue = queue
        self.model_storage = model_storage
        self.warmstart_steps = warmstart_steps
        self.worker_weight_ckpt_interval = worker_weight_ckpt_interval
        self.queue_high_water = queue_high_water
        self.queue_low_water = queue_low_water
        self.last_model_index = -1

        self.env = env_config.spawn(seed=rank)

        actor_net = ActorNetwork(hidden_dim, depth, action_dim)
        dummy_obs = jnp.ones((1, state_dim))
        key = jax.random.PRNGKey(rank)
        self.key = key
        key, init_key = jax.random.split(key)
        self.local_actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(init_key, dummy_obs),
            tx=optax.set_to_zero(),
        )

    def _check_and_sync_weights(self, trained_steps):
        """Pull actor params from ModelSharedStorage if a new version is available."""
        new_model_index = trained_steps // self.worker_weight_ckpt_interval
        if new_model_index > self.last_model_index:
            self.last_model_index = new_model_index
            weights = ray.get(self.model_storage.get_weights.remote())
            if weights is not None:
                self.local_actor = self.local_actor.replace(
                    params=weights['actor_params']
                )

    def _wait_for_queue_space(self):
        """Back off if queue is too full, resume when it drains."""
        if self.queue.qsize() >= self.queue_high_water:
            while self.queue.qsize() > self.queue_low_water:
                time.sleep(1.0)

    def run(self):
        """Main loop: step env, push transitions to queue."""

        obs, _ = self.env.reset()
        warmstart_done = False
        while True:
            trained_steps = ray.get(self.model_storage.get_counter.remote())
            if not warmstart_done:
                warmstart_done = ray.get(self.model_storage.get_warmstart_signal.remote())

            self._check_and_sync_weights(trained_steps)

            if not warmstart_done:
                actions = self.env.action_space.sample()
            else:
                self.key, action_key = jax.random.split(self.key)
                actions, _ = _select_actions(self.local_actor, obs, action_key)
                actions = np.asarray(actions)

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            dones = np.logical_or(terminations, truncations)

            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(
                    dones[:, None], np.stack(infos["final_obs"]), next_obs
                )

            self._wait_for_queue_space()
            self.queue.put((obs, actions, rewards, buffer_obs, dones))
            obs = next_obs


class ParallelReplayOrchestrator(Orchestrator):
    """Orchestrator for parallel replay with online RL training.

    Decouples data collection from training: multiple DataWorker Ray actors
    interact with environments in parallel, pushing transitions into a shared
    Ray Queue. A background daemon thread drains the queue into the local
    ParallelReplayBuffer. The main process runs the training loop, sampling
    from the buffer and updating the agent.

    TODO: model checkpointing every X iterations, validation/evaluation every
    Z iterations, and structured logging — to be organized modularly.
    """
    def __init__(
        self,
        agent,
        data_config,
        num_epochs=None,
        env_config=None,
        warmstart_steps=None,
        num_workers=10,
        worker_weight_ckpt_interval=100,
        queue_maxsize=1000,
        drain_interval=1.0,
        drain_threshold=64,
    ):
        self.agent = agent
        self.data_config = data_config
        self.num_epochs = num_epochs
        self.warmstart_steps = warmstart_steps
        self.num_steps_per_epoch = getattr(data_config, 'NUM_STEPS_PER_EPOCH', None)
        self.num_workers = num_workers
        self.worker_weight_ckpt_interval = worker_weight_ckpt_interval
        self.env_config = env_config

        self.data_collection = ParallelReplayBuffer(
            data_config,
            maxsize=queue_maxsize,
            drain_interval=drain_interval,
            drain_threshold=drain_threshold,
        )
        self.model_storage = ModelSharedStorage.remote()

    def _launch_workers(self):
        self.workers = []
        for rank in range(self.num_workers):
            worker = DataWorker.remote(
                rank=rank,
                queue=self.data_collection.queue,
                model_storage=self.model_storage,
                env_config=self.env_config,
                state_dim=self.agent.state_dim,
                action_dim=self.agent.action_dim,
                hidden_dim=self.agent.hidden_dim,
                depth=self.agent.depth,
                warmstart_steps=self.warmstart_steps,
                worker_weight_ckpt_interval=self.worker_weight_ckpt_interval,
            )
            self.workers.append(worker)
        self.worker_tasks = [w.run.remote() for w in self.workers]

    def start_online(self):
        start_time = time.time()
        print('-'*10 + 'starting parallel replay with online RL training' + '-'*10)
        print(f'Workers: {self.num_workers}, '
              f'worker_weight_ckpt_interval: {self.worker_weight_ckpt_interval}')

        self._launch_workers()
        self.data_collection.start_drain()

        # Wait for enough warmstart transitions before training
        print('Collecting warmstart transitions...')
        while len(self.data_collection.dataset) < self.warmstart_steps:
            time.sleep(1)

        self.model_storage.set_warmstart_signal.remote()
        self.data_collection.create_train_loader()
        print(f'Warmstart complete. Buffer size: {len(self.data_collection.dataset)}. '
              f'Beginning training...')

        # Push initial weights so workers can sync
        self.model_storage.set_weights.remote({
            'actor_params': self.agent.actor.params,
        })

        global_step = 0
        for epoch in range(self.num_epochs):
            for step in range(self.num_steps_per_epoch):
                batch = self.data_collection.sample_batch()
                # Flatten (batch, seq_len, num_tasks, feat_dim) → (N, feat_dim)
                # e.g. states: (32,1,10,59)→(320,59), rewards: (32,1,10)→(320,)
                batch = tuple(
                    b.reshape(-1, *b.shape[3:]) if b.dim() > 2 else b.reshape(-1)
                    for b in batch
                )
                self.agent.update(*batch)

                global_step += 1
                self.model_storage.incr_counter.remote()

                if global_step % self.worker_weight_ckpt_interval == 0:
                    self.model_storage.set_weights.remote({
                        'actor_params': self.agent.actor.params,
                    })

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"global_step={global_step}, "
                  f"buffer={len(self.data_collection.dataset)}, "
                  f"time={elapsed/60:.1f}min")

        print(f"Parallel training complete in {(time.time()-start_time)/60:.1f}min")
