import unittest
import numpy as np
from agent import Agent


class AgentTest(unittest.TestCase):
  def test_agent_init(self):
    agent = Agent((6, 1))
    self.assertIsNotNone(agent)

  def test_agent_output_has_correct_type(self):
    agent = Agent((6,1))
    observation = np.array([])
    action = agent.act(observation)
    self.assertIsInstance(action, np.ndarray)


if __name__ == "__main__":
  unittest.main()