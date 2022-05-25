import gym
import torch
from torch.optim import Adam
from torch import nn
from pathlib import Path

from src.agent import Agent

from src.loss_functions import update_models
from src.return_estimator import DiscountReturn
from src.advantage import GAE
from src.trajectory_buffer import TrajectoryReplayBuffer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_FREQ = 10
    ### Setting up Hyperparameters ###
    EPISODES = 50
    EPISODE_LEN = 100
    LEARNING_RATE = 0.0001

    OBS_DIM = 3
    ACT_DIM = 1
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
    ### Init Agent ###
    agent = Agent(
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        hidden_dim=HIDDEN_DIM,
        activation=nn.Softmax(dim=-1),
        save_path=Path("./assets"),
    )
    agent.to(device)
    ### Init Agent ###

    ### Init Optimizer ###
    pi_optim = Adam(agent.policy.parameters(), lr=LEARNING_RATE)
    val_optim = Adam(agent.value_func.parameters(), lr=LEARNING_RATE)
    ### Init Optimizer ###

    ### Training Loop ###
    env = gym.make("Pendulum-v1")
    env = gym.wrappers.FlattenObservation(env)
    for e in range(EPISODES):
        obs = env.reset()
        for t in range(EPISODE_LEN):
            env.render()

            obs_cuda_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
            act, value, mean_act = agent.step(obs_cuda_tensor)
            obs, reward, done, _ = env.step(act)
            trajectory_buffer.store(t, obs, act, value, reward, mean_act)
            
            if done:
                print(f"Episode finished after {t} timesteps")
                break

        trajectory_buffer.finish_trajectory()
        data = trajectory_buffer.get_trajectories()
        pi_loss, v_loss = update_models(agent, data, pi_optim, val_optim, 10)

    env.close()
