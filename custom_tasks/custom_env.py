from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from custom_tasks.move_arm_only_action_mode import MoveArmOnly


def get_env_task_env(headless: bool = True):
    env = Environment(
        action_mode=MoveArmOnly(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        headless=headless,
    )
    env.launch()
    task = env.get_task(ReachTarget)
    return env, task
