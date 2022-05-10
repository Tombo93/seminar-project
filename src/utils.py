from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from rlbench.backend.observation import Observation


class ReturnEstimator(ABC):
    @abstractmethod
    def get_return(self, rewards: np.ndarray) -> np.float:
        raise NotImplementedError


@dataclass
class DiscountReturn(ReturnEstimator):
    gamma: Optional[float] = 0.99

    def get_return(self, rewards: np.ndarray) -> np.float:
        pot = np.cumsum(np.ones(len(rewards))) - 1
        g = np.full(len(pot), fill_value=self.gamma)
        discount_gamma = g**pot
        return np.sum(rewards * discount_gamma)


class TrajectoryReplayBuffer:
    """A buffer class for storing trajectory data"""

    def __init__(self, return_estimator: ReturnEstimator, buf_size: int = 20) -> None:
        self._buf_size = buf_size
        self._r = return_estimator
        self._act = np.zeros(buf_size, dtype=np.float)
        self._mean_act = np.zeros(buf_size, dtype=np.float)
        self._v = np.zeros(buf_size, dtype=np.float)
        self._q = np.zeros(buf_size, dtype=np.float)

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
        self._v[idx] = value
        self._q[idx] = reward
        self._mean_act[idx] = mean_act

    def compute_advantage(self):
        return self._r.get_return(self._v)

    def get_trajectories(self):
        data = dict(Q=self.Q, V=self.V)
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
