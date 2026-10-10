"""MT50 multi-task SAC with the MOORE network.

Replicates metaworld-algorithms/examples/multi_task/moore_mt50.py:

    Run("mt50_moore"):
      env:           MetaworldConfig(env_id="MT50", terminate_on_success=False)
      algorithm:     MTSAC, num_tasks=50, gamma=0.99, num_critics=2
        actor:       MOOREConfig (width=400, depth=3, num_experts=6),
                     log_std_min=-20, log_std_max=2, lr=3e-4, max_grad_norm=1.0
        critic:      MOOREConfig (width=400, depth=3, num_experts=6),
                     lr=3e-4, max_grad_norm=1.0
        temperature: lr=1e-4 (no grad clipping)
      training:      total_steps=1e8, buffer_size=5e6, batch_size=6400

Geodesic does not yet expose total_steps directly; the orchestrator step
budget is num_epochs * NUM_STEPS_PER_EPOCH. Defaults below give 1e8 total steps
(1000 epochs * 100k steps). Override via tyro CLI flags for shorter local runs.

Usage:
    PYTHONPATH=. uv run python projects/grin/moore_metaworld_mt50.py
    PYTHONPATH=. uv run python projects/grin/moore_metaworld_mt50.py \\
        --total-steps 1000000 --num-epochs 10
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")

from envs import MetaworldConfig
from jax_agents import MooreSACAgent
from orchestrator import Orchestrator


# ---- Reference hyperparameters from moore_mt50.py ----
NUM_TASKS = 50
GAMMA = 0.99
NUM_EXPERTS = 6           # MOOREConfig default; same as MT50 reference
HIDDEN_DIM = 400          # NeuralNetworkConfig.width default
DEPTH = 3                 # NeuralNetworkConfig.depth default
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 1e-4
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 6400         # transitions per gradient step (post-flatten)
BUFFER_SIZE = 5_000_000


@dataclass(frozen=True)
class Args:
    seed: int
    total_steps: int
    num_epochs: int
    warmstart_steps: int
    data_dir: Path
    track: bool
    resume: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="MT50 MOORE multi-task SAC")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=int(1e8))
    p.add_argument("--num-epochs", type=int, default=1000,
                   help="NUM_STEPS_PER_EPOCH = total_steps // num_epochs")
    p.add_argument("--warmstart-steps", type=int, default=1500,
                   help="~30 episodes * 50 tasks (per MTSAC convention)")
    p.add_argument("--data-dir", type=Path, default=Path("./run_results"))
    p.add_argument("--track", action="store_true",
                   help="enable wandb tracking — not wired in geodesic yet")
    p.add_argument("--resume", action="store_true",
                   help="resume — not wired in geodesic yet")
    a = p.parse_args()
    return Args(seed=a.seed, total_steps=a.total_steps, num_epochs=a.num_epochs,
                warmstart_steps=a.warmstart_steps, data_dir=a.data_dir,
                track=a.track, resume=a.resume)


class MT50OnlineConfig:
    """Mirrors moore_mt50.py training_config."""
    RL_MODE = "ONLINE"
    SEQ_LEN = 1
    STRIDE = 1
    BUFFER_CAPACITY = BUFFER_SIZE
    # Orchestrator flattens (batch, seq_len, num_tasks, feat) -> (N, feat),
    # so dataloader batch_size = BATCH_SIZE // num_tasks gives BATCH_SIZE
    # transitions per gradient step (matches reference).
    TRAIN_BATCH_SIZE = BATCH_SIZE // NUM_TASKS  # 6400 // 50 = 128
    SHUFFLE = False
    NUM_STEPS_PER_EPOCH = 100_000        # overridden in main() to match args.total_steps / args.num_epochs
    SEED = 1                             # overridden in main()


def main() -> None:
    args = parse_args()

    env_config = MetaworldConfig(env_id="MT50", terminate_on_success=False)
    state_dim = env_config.observation_space.shape[0]
    action_dim = int(env_config.action_space.shape[0])

    agent = MooreSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=NUM_TASKS,
        num_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        gamma=GAMMA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        alpha_lr=ALPHA_LR,
        max_grad_norm=MAX_GRAD_NORM,
        seed=args.seed,
    )

    data_config = MT50OnlineConfig()
    data_config.NUM_STEPS_PER_EPOCH = max(1, args.total_steps // args.num_epochs)
    data_config.SEED = args.seed

    orc = Orchestrator(
        agent=agent,
        data_config=data_config,
        num_epochs=args.num_epochs,
        env_config=env_config,
        warmstart_steps=args.warmstart_steps,
    )

    if args.track:
        print("[note] --track ignored: wandb is not yet wired in geodesic.")
    if args.resume:
        print("[note] --resume ignored: resume is not yet wired in geodesic's online loop.")

    print(
        f"Starting MT50 MOORE run | seed={args.seed} | "
        f"total_steps={args.total_steps:,} | "
        f"epochs={args.num_epochs} x steps/epoch={data_config.NUM_STEPS_PER_EPOCH:,} | "
        f"warmstart={args.warmstart_steps} | batch={BATCH_SIZE} | "
        f"buffer={BUFFER_SIZE:,}"
    )
    orc.start_online()


if __name__ == "__main__":
    main()
