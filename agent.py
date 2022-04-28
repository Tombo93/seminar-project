from typing import Tuple
import numpy as np

from rlbench.backend.observation import Observation


class Agent:
  def __init__(self, action_shape: Tuple[int]) -> None:
      self.action_shape = action_shape

  def act(self, obs: Observation) -> np.ndarray:
    """TODO: Implement simple network using a singular input from Observation
    """
    arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
    gripper = [1.0]  # Always open
    return np.concatenate([arm, gripper], axis=-1)
