from typing import Any, Tuple
import numpy as np


Action = Any


class Agent:
  def __init__(self, action_shape: Tuple[int]) -> None:
      self.action_shape = action_shape

  def act(self, observation: np.ndarray) -> Action:
    """
    Here the policy learning network
    that learns to generate coordinates for the arm
    """
    arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
    gripper = [1.0]  # Always open
    return np.concatenate([arm, gripper], axis=-1)
