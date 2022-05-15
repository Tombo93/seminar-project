from dataclasses import dataclass
from typing import Protocol
import numpy as np


class ReturnEstimator(Protocol):
    def get_return(self, rewards: np.ndarray) -> np.ndarray:
        """Calculates a return over a trajectory"""
        raise NotImplementedError


@dataclass
class DiscountReturn:
    """Calculate the discounted return over a trajectory, with discount factor gamma."""
    gamma: float = 0.99

    def get_return(self, rewards: np.ndarray) -> np.ndarray:
        pot = np.cumsum(np.ones(len(rewards))) - 1
        g = np.full(len(pot), fill_value=self.gamma)
        discount_gamma = g**pot
        return rewards * discount_gamma
