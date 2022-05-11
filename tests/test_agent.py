import unittest
import numpy as np
from src.utils import DiscountReturn, TrajectoryReplayBuffer, GAE


class AgentTest(unittest.TestCase):
    def test_discounted_return(self):
        d = DiscountReturn()
        r = d.get_return(np.ones(10))
        self.assertAlmostEqual(r, 9.56179, places=5)

    def test_gae(self):
        rewards = np.zeros(10)
        val_old = np.ones(10)
        val_new = np.ones(10)
        done = np.ones(10)
        gae = GAE(rewards, val_old, val_new, done)
        adv, _ = gae.estimate()
        self.assertEqual(adv, np.ones(10))

    def test_advantage_computation(self):
        episode_len = 10
        r_estimator = DiscountReturn(gamma=0.99)
        buf = TrajectoryReplayBuffer(r_estimator, 6, 1, episode_len)
        for idx in range(episode_len):
            buf.store(idx, 1, 0.5, 0, 1)
        advantage = buf.compute_advantage()
        self.assertAlmostEqual(advantage, 47.808, places=2)


if __name__ == "__main__":
    unittest.main()
