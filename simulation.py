from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from agent import Agent
from move_arm_only_action_mode import MoveArmOnly


def main() -> None:
    env = Environment(
        action_mode=MoveArmOnly(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        headless=True,
    )
    env.launch()
    task = env.get_task(ReachTarget)
    agent = Agent(env.action_shape)
    training_steps = 120
    episode_length = 40
    obs = None
    for i in range(training_steps):
        if i % episode_length == 0:
            print("Reset Episode")
            descriptions, obs = task.reset()
            print(descriptions)
        action = agent.act(obs)
        print(action)
        obs, reward, terminate = task.step(action)
    print("Done")
    env.shutdown()


if __name__ == "__main__":
    main()
