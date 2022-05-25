import numpy as np
import torch

from src.return_estimator import ReturnEstimator
from src.advantage import Advantage


class TrajectoryReplayBuffer:
    """A buffer for storing trajectory data"""

    def __init__(
        self,
        device: torch.device,
        return_estimator: ReturnEstimator,
        advantage: Advantage,
        buf_size: int,
        obs_dim: int,
        act_dim: int,
    ) -> None:
        self._device = device
        self._ret_estimator = return_estimator
        self._adv_estimator = advantage
        self._buf_size = buf_size
        self._obs = np.zeros((buf_size, obs_dim), dtype=float)
        self._act = np.zeros((buf_size, act_dim), dtype=float)
        self._val = np.zeros(buf_size, dtype=float)
        self._rewards = np.zeros(buf_size, dtype=float)
        self._mean_act = np.zeros(buf_size, dtype=float)

    def store(
        self,
        idx: int,
        obs: np.ndarray,
        action: np.ndarray,
        value: np.ndarray,
        reward: float,
        mean_act: float,
    ) -> None:
        assert idx < self._buf_size
        self._obs[idx] = obs
        self._act[idx] = action
        self._val[idx] = value
        self._rewards[idx] = reward
        self._mean_act[idx] = mean_act

    def finish_trajectory(self):
        self._ret = self._ret_estimator.get_return(self._rewards)
        self._adv = self._adv_estimator.estimate(self._rewards, self._val)

    def get_trajectories(self):
        data = dict(
            obs=self._obs,
            act=self._act,
            val=self._val,
            ret=self._ret,
            logp=self._mean_act,
            adv=self._adv,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(self._device)
            for k, v in data.items()
        }
