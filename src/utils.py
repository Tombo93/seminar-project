from typing import Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from rlbench.backend.observation import Observation


class ReturnEstimator(ABC):
    @abstractmethod
    def get_return(self, rewards: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Advantage(ABC):
    @abstractmethod
    def estimate(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DiscountReturn(ReturnEstimator):
    gamma: Optional[float] = 0.99

    def get_return(self, rewards: np.ndarray) -> np.ndarray:
        pot = np.cumsum(np.ones(len(rewards))) - 1
        g = np.full(len(pot), fill_value=self.gamma)
        discount_gamma = g**pot
        return rewards * discount_gamma


@dataclass
class GAE(Advantage):
    rewards: np.ndarray
    values: np.ndarray
    batch_size: int = rewards.shape(0)
    adv: np.ndarray = np.zeros(batch_size)
    returns: np.ndarray() = np.zeros(batch_size)
    lamda: Optional[float] = 0.5
    gamma: Optional[float] = 0.99
    discount_return: ReturnEstimator = DiscountReturn(gamma=lamda * gamma)

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        rew = np.append(self.rewards, 0)
        val = np.append(self.values, 0)
        deltas = rew[:-1] + (self.gamma * val[1:]) - val[:-1]
        self.adv = self.discount_return.get_return(deltas)
        self.discount_return.gamma = self.gamma
        self.returns = self.discount_return.get_return(rew)[:-1] # value function targets
        return self.adv, self.returns


class TrajectoryReplayBuffer:
    """A buffer class for storing trajectory data"""

    def __init__(
        self,
        advantage: Advantage,
        obs_dim: int,
        act_dim: int,
        val_dim: int,
        buf_size: int = 20,
    ) -> None:
        self._buf_size = buf_size
        self._adv_estimator = advantage
        self._obs = np.zeros((buf_size, obs_dim), dtype=np.float)
        self._act = np.zeros((buf_size, act_dim), dtype=np.float)
        self._val = np.zeros((buf_size, val_dim), dtype=np.float)
        self._adv = np.zeros(buf_size, dtype=np.float)
        self._mean_act = np.zeros(buf_size, dtype=np.float)
        self._rewards = np.zeros(buf_size, dtype=np.float)

    def store(
        self,
        idx: int,
        action: np.float,
        value: np.float,
        reward: np.float,
        mean_act: np.float,
    ) -> None:
        assert idx < self._buf_size
        self._act[idx] = action
        self._val[idx] = value
        self._rewards[idx] = reward
        self._mean_act[idx] = mean_act

    def compute_advantage(self):
        adv_ = self._adv_estimator(self._rewards, self._val)
        self._adv, val_func_targets = adv_.estimate()
        return self._adv, val_func_targets

    def expected_returns(self, arr: np.ndarray) -> np.ndarray:
        expected_returns = np.zeros(arr.shape)
        for i in reversed(range(len(arr))):
            ret_t = self._return_estimator(arr[i:])
            expected_returns[i] = ret_t
        return expected_returns

    def get_trajectories(self):
        data = dict(V=self._val)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def plot_img(img: np.ndarray) -> None:
    plt.imshow(img)
    plt.show()


def save_img(fname: str, img: np.ndarray) -> None:
    plt.imsave(fname, img, format="png")


def save_observation_imgs(obs: Observation, save_folder: str = "assets") -> None:
    for k, v in obs.__dict__.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.uint8:
                v = v.astype(float) / 225
            fname = os.path.join(save_folder, f"{k}.png")
            save_img(fname, v)
