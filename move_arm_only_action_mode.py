import numpy as np
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene


class MoveArmOnly(ActionMode):
  def action(self, scene: Scene, action: np.ndarray):
    arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
    arm_action = np.array(action[:arm_act_size])
    ee_action = np.array(action[arm_act_size:])
    self.arm_action_mode.action(scene, arm_action)
    self.gripper_action_mode.action(scene, ee_action)

  def action_shape(self, scene: Scene):
    return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
      self.gripper_action_mode.action_shape(scene))
