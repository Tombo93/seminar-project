import unittest
import numpy as np
from src.utils import DiscountReturn


class AgentTest(unittest.TestCase):
    def test_discounted_return(self):
        d = DiscountReturn()
        r = d.get_return(np.ones(10))
        self.assertAlmostEqual(r, 9.56179, places=5)


if __name__ == "__main__":
    unittest.main()
