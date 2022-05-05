import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rlbench.backend.observation import Observation


class DiagonalGaussian(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    hidden_dim: int,
    action_dim: int,
    activation
  ) -> None:
      super(DiagonalGaussian, self).__init__()
      log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
      self.covariance_matrix = torch.nn.Parameter(torch.as_tensor(log_std))
      self.mean_action_net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, action_dim),
        activation())

  def _distribution(self, observation):
      mean_act = self.mean_action_net(observation)
      covariance_mat = torch.exp(self.covariance_matrix)
      return Normal(mean_act, covariance_mat)

  def _log_probs_from_dist(self, policy_dist, action):
      return policy_dist.log_prob(action).sum(axis=-1)

  def forward(self, observation, action=None):
      policy_dist = self._distribution(observation)
      logp_act = None
      if action is not None:
          logp_act = self._log_probs_from_dist(policy_dist, action)
      return policy_dist, logp_act


class ValueFunctionLearner(nn.Module):
  def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, activation) -> None:
      super(ValueFunctionLearner, self).__init__()
      self.v_net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, action_dim),
        activation()
      )

  def forward(self, observation):
      # return torch.squeeze(self.v_net(observation), -1)
      return self.v_net(observation)


class Agent(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int=32,
    activation=nn.Softmax
  ) -> None:
    self.policy = DiagonalGaussian(obs_dim, hidden_dim, action_dim, activation)
    self.value_func = ValueFunctionLearner(obs_dim, hidden_dim, action_dim, activation)
  
  def step(self, obs: Observation):
    with torch.no_grad():
      policy_dist = self.policy._distribution(obs)
      action = policy_dist.sample()
      mean_action = self.policy._log_probs_from_dist(policy_dist, action)
      value = self.value_func(obs)
    return action.numpy(), value.numpy(), mean_action.numpy()

  def act(self, obs: Observation):
    return self.step(obs)[0]