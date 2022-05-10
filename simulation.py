from rlbench.environment import Environment, TaskEnvironment

import torch
from torch import nn
from torch.optim import Adam

from src.agent import Agent
from src.utils import TrajectoryReplayBuffer, DiscountReturn
from custom_tasks.custom_env import get_env_task_env


def main(
    env: Environment,
    task: TaskEnvironment,
    trajectory_buf: TrajectoryReplayBuffer,
    episodes: int = 2,
    episode_length: int = 40,
    learning_rate: float = 0.0001,
) -> None:

    _, obs = task.reset()

    agent = Agent(
        obs_dim=obs.get_low_dim_data().shape[0],
        action_dim=env.action_shape[0],
        activation=nn.Softmax(dim=-1),
    )
    pi_optim = Adam(agent.policy.parameters(), lr=learning_rate)
    val_optim = Adam(agent.value_func.parameters(), lr=learning_rate)

    trajectories = []
    for _ in range(episodes):
        print("Reset Episode")
        _, obs = task.reset()
        for t in range(episode_length):

            action, value, mean_action = agent.step(
                torch.as_tensor(obs.get_low_dim_data(), dtype=torch.float32)
            )

            print(action)
            obs, reward, done = task.step(action)

            trajectory_buf.store(t, action, value, reward, mean_action)
            
            pi_optim.zero_grad()
            val_optim.zero_grad()
            # policy_loss = compute_pi_loss()
            # policy_loss.backward()
            # val_loss = compute_val_loss()
            # val_loss.backward()
            pi_optim.step()
            val_optim.step()
        trajectories.append(trajectory_buf.get_trajectories())

    print("Done")
    env.shutdown()


if __name__ == "__main__":
    episode_len, obs_dim, act_dim = 40, 29, 8
    buf = TrajectoryReplayBuffer(
        DiscountReturn(), obs_dim, act_dim, buf_size=episode_len
    )
    env, task = get_env_task_env()
    main(env, task, buf, episode_length=episode_len)
