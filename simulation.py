from rlbench.environment import Environment, TaskEnvironment

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.agent import Agent
from src.loss_functions import update_models
from src.return_estimator import DiscountReturn
from src.advantage import GAE
from src.trajectory_buffer import TrajectoryReplayBuffer
from custom_tasks.custom_env import get_env_task_env


def main(
    env: Environment,
    task: TaskEnvironment,
    trajectory_buf: TrajectoryReplayBuffer,
    device: str,
    episodes: int = 2,
    episode_length: int = 40,
    learning_rate: float = 0.0001,
    safe_freq=10,
    load_model=False
) -> None:
    writer = SummaryWriter()

    _, obs = task.reset()

    agent = Agent(
        obs_dim=obs.get_low_dim_data().shape[0],
        action_dim=env.action_shape[0],
        activation=nn.Softmax(dim=-1),
        save_path=Path("./models/reach-target"),
    )
    if load_model:
        agent.load_model(eval=False, device=device)
    agent.to(device)

    pi_optim = Adam(agent.policy.parameters(), lr=learning_rate)
    val_optim = Adam(agent.value_func.parameters(), lr=learning_rate)

    for e in range(episodes):
        print("Reset Episode")
        _, obs = task.reset()
        for t in range(episode_length):

            obs_tensor = torch.as_tensor(
                obs.get_low_dim_data(), dtype=torch.float32
            ).to(device)
            action, value, mean_action = agent.step(obs_tensor)
            obs, reward, done = task.step(action)

            trajectory_buf.store(
                t, obs.get_low_dim_data(), action, value, reward, mean_action
            )

            writer.add_scalar("cumulative reward", reward, (e+1)*(t+1))
            writer.add_scalar("state value", value, (e+1)*(t+1))

            if done:
                print(f"Episode finished after {t} timesteps")
                break

        trajectory_buf.finish_trajectory()
        trajectory_data = trajectory_buf.get_trajectories()
        update_models(agent, trajectory_data, pi_optim, val_optim, 1, logger=None)

        if (e % safe_freq == 0) or (e == episodes-1):
            agent.save_model()

    print("Done")
    writer.close()
    env.shutdown()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SAFE_FREQ = 10
    # Hyperparameters #
    EPISODES = 1
    EPISODE_LEN = 10
    LEARNING_RATE = 0.0001

    OBS_DIM = 29
    ACT_DIM = 8
    HIDDEN_DIM = 32
    BUF_SIZE = EPISODE_LEN

    GAMMA, LAMDA = 0.99, 0.5
    # --------------- #
    return_estimator = DiscountReturn(gamma=GAMMA)
    advantage_return = DiscountReturn(gamma=GAMMA * LAMDA)
    advantage = GAE(advantage_return, lamda=LAMDA, gamma=GAMMA)

    trajectory_buffer = TrajectoryReplayBuffer(
        device,
        return_estimator,
        advantage,
        buf_size=BUF_SIZE,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
    )

    env, task = get_env_task_env(headless=True)

    main(
        env=env,
        task=task,
        trajectory_buf=trajectory_buffer,
        device=device,
        episodes=EPISODES,
        episode_length=EPISODE_LEN,
        learning_rate=LEARNING_RATE,
        safe_freq=SAFE_FREQ,
        load_model=True
    )
