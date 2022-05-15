from typing import Protocol
from dataclasses import dataclass
import numpy as np

from src.return_estimator import ReturnEstimator


class Advantage(Protocol):
    def estimate(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Calculates an advantage."""
        raise NotImplementedError


@dataclass
class GAE:
    return_estimator: ReturnEstimator
    lamda: float = 0.5
    gamma: float = 0.99

    def estimate(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        rew = np.append(rewards, 0)
        val = np.append(values, 0)
        deltas = rew[:-1] + (self.gamma * val[1:]) - val[:-1]
        adv = self.return_estimator.get_return(deltas)
        return adv
