import unittest
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class EnvironmentTest(unittest.TestCase):
    def test_env_init(self):
        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfig(),
            headless=True,
        )
        self.assertIsNotNone(env)


if __name__ == "__main__":
    unittest.main()
