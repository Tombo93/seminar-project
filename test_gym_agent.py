import gym
import torch
from torch import nn
from pathlib import Path

from src.agent import Agent


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### Setting up Hyperparameters ###
    EPISODES = 50
    EPISODE_LEN = 4000
    
    OBS_DIM = 3
    ACT_DIM = 1
    HIDDEN_DIM = 64
    ### -------------------------- ###

    ### Init Agent ###
    agent = Agent(
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        hidden_dim=HIDDEN_DIM,
        activation=nn.Softmax(dim=-1),
        save_path=Path("./models"),
    )
    agent.load_model(eval=True, device=device)
    agent.to(device)
    ### ---------- ###
    
    ### Init env ###
    env = gym.make("Pendulum-v1")
    env = gym.wrappers.FlattenObservation(env)
    ### -------- ###
    
    for e in range(EPISODES):
        obs = env.reset()
        for t in range(EPISODE_LEN):
            env.render()
            obs_cuda_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
            act = agent.act(obs_cuda_tensor)
            # print(f"action: {act}\nvalue: {value}\nmean action: {mean_act}")
            obs, reward, done, _ = env.step(act)
    
    env.close()
    