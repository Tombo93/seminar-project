import gym
import torch
from torch import nn
from torch.optim import Adam
from pathlib import Path

from src.advantage import GAE
from src.return_estimator import DiscountReturn
from src.trajectory_buffer import TrajectoryReplayBuffer
from src.agent import Agent
from src.loss_functions import update_models


device = 'cpu'
SAVE_FREQ = 10
### Setting up Hyperparameters ###
EPISODES = 50
EPISODE_LEN = 4000
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
    save_path=Path("./tmp")
)
agent.load_model('/dependencies')
agent.to(device)
### Init Agent ###

### Init Optimizer ###
pi_optim = Adam(agent.policy.parameters(), lr=LEARNING_RATE)
val_optim = Adam(agent.value_func.parameters(), lr=LEARNING_RATE)
### Init Optimizer ###

env = gym.make("Pendulum-v0")
env = gym.wrappers.FlattenObservation(env)
for e in range(EPISODES): 
    obs = env.reset()
    for t in range(EPISODE_LEN):
      obs_cuda_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
      act, value, mean_act = agent.step(obs_cuda_tensor)
      # print(f"action: {act}\nvalue: {value}\nmean action: {mean_act}")
      obs, reward, done, _ = env.step(act)
      # print(f"observation: {obs}\nreward: {reward}\ndone: {done}")
      # print("-------------------")
      trajectory_buffer.store(t, obs, act, value, reward, mean_act)
      if done:
          print(f"Episode finished after {t} timesteps")
          break

    trajectory_buffer.finish_trajectory()
    # print(f"advantage: {trajectory_buffer._adv},
    # value targets(estimated return): {trajectory_buffer._ret}")
    data = trajectory_buffer.get_trajectories()

    pi_loss, v_loss = update_models(agent, data, 80)

    if (e % SAVE_FREQ == 0) or (e == EPISODES-1):
        agent.save_model()

env.close()