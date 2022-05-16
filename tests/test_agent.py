import unittest
import numpy as np
from src.return_estimator import DiscountReturn
from src.advantage import GAE
from src.trajectory_buffer import TrajectoryReplayBuffer


class AgentTest(unittest.TestCase):
    def test_policy_loss(self):
        pass
    
    def test_discounted_return(self):
        d = DiscountReturn(gamma=0.99)
        rewards = np.ones(10)
        r = d.get_return(rewards)
        self.assertEqual(r, np.ones(10))

    def test_gae(self):
        rewards = np.zeros(10)
        values = np.ones(10)
        return_estimator = DiscountReturn()
        gae = GAE(return_estimator)
        adv, _ = gae.estimate(rewards, values)
        self.assertEqual(adv, np.ones(10))

    def test_advantage_computation(self):
        episode_len = 10
        r_estimator = DiscountReturn(gamma=0.99)
        advantage = GAE(r_estimator)
        buf = TrajectoryReplayBuffer(r_estimator, advantage, episode_len, 1, 1, 1)
        for idx in range(episode_len):
            buf.store(idx, np.array([0.5]), np.array([0]), np.array([0]), 1.0, 0.5)
        buf.finish_trajectory()
        advantage = buf.get_trajectories()
        self.assertAlmostEqual(advantage, 47.808, places=2)


if __name__ == "__main__":
    unittest.main()
