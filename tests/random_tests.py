#!/usr/bin/env python
"""
Test code for the random generator
"""

import unittest
from soprano.rnd import Random


class TestRandom(unittest.TestCase):
    def test_randomness(self):

        self.assertFalse(Random.random() == Random.random())

    def test_seed(self):

        Random.reseed(12345)
        i1 = Random.randint(1000)
        Random.reseed(12345)
        i2 = Random.randint(1000)

        self.assertEqual(i1, i2)


if __name__ == "__main__":
    unittest.main()
