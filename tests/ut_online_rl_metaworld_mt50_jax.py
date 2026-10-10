import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting.*")

from jax_agents import JAXSACAgent
from orchestrator import Orchestrator
from envs import MetaworldConfig
import numpy as np

class MT50OnlineConfig:
    RL_MODE = "ONLINE"
    SEQ_LEN = 1
    STRIDE = 1
    BUFFER_CAPACITY = 100000
    TRAIN_BATCH_SIZE = 32
    SHUFFLE = False
    NUM_STEPS_PER_EPOCH = 500
    SEED = 42

def ut_mt50_online_jax():
    env_config = MetaworldConfig(env_id="MT50", terminate_on_success=False)
    state_dim = env_config.observation_space.shape[0]   # 59 (49 state + 10 one-hot)
    action_dim = int(np.prod(env_config.action_space.shape))  # 4
    agent = JAXSACAgent(state_dim=state_dim, action_dim=action_dim)
    data_config = MT50OnlineConfig()
    orc = Orchestrator(
        agent=agent,
        data_config=data_config,
        num_epochs=10,
        env_config=env_config,
        warmstart_steps=100,
    )
    orc.start_online()

if __name__ == "__main__":
    ut_mt50_online_jax()
