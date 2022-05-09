from tkinter.tix import Tree
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

import torch
from torch.optim import Adam
from torch import nn

from src.agent import Agent
from custom_tasks.move_arm_only_action_mode import MoveArmOnly


def main(
    episodes: int = 80, episode_length: int = 40, learning_rate: float = 0.0001
) -> None:

    env = Environment(
        action_mode=MoveArmOnly(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        headless=True,
    )
    env.launch()
    task = env.get_task(ReachTarget)
    _, obs = task.reset()

    agent = Agent(
        obs_dim=obs.get_low_dim_data().shape[0],
        action_dim=env.action_shape[0],
        activation=nn.Softmax(dim=-1),
    )
    pi_optim = Adam(agent.policy.parameters(), lr=learning_rate)
    val_optim = Adam(agent.value_func.parameters(), lr=learning_rate)

    obs = torch.tensor(obs.get_low_dim_data(), dtype=torch.float32)
    for i in range(episodes):
        if i % episode_length == 0:
            print("Reset Episode")
            descriptions, obs = task.reset()
            obs = torch.tensor(obs.get_low_dim_data(), dtype=torch.float32)
            print(descriptions)
        action, value, mean_action = agent.step(obs)
        print(action)
        obs, reward, done = task.step(action)
        obs = torch.tensor(obs.get_low_dim_data(), dtype=torch.float32)

        pi_optim.zero_grad()
        val_optim.zero_grad()
        # policy_loss = compute_pi_loss()
        # policy_loss.backward()
        # val_loss = compute_val_loss()
        # val_loss.backward()
        pi_optim.step()
        val_optim.step()

    print("Done")
    env.shutdown()


if __name__ == "__main__":
    main()
