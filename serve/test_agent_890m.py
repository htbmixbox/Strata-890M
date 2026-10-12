import unittest
from serve.agent_890m import cached_prefix, next_waiter

class Agent890MTests(unittest.TestCase):
    def test_prefix_resumption(self):
        self.assertEqual(cached_prefix([1, 2, 3, 4, 5], [[1, 2], [1, 2, 3, 4], [9, 8]]), 4)
        self.assertEqual(cached_prefix([1, 2], [[1, 2]]), 0)
        self.assertEqual(cached_prefix([1, 2, 3], [[4, 5]]), 0)

    def test_cached_continuation_first_but_no_starvation(self):
        entries = [[14000, 100.0, 1, 8000], [15000, 100.2, 2, 64], [500, 100.3, 3, 500]]
        self.assertIs(next_waiter(entries, 101), entries[1])
        self.assertIs(next_waiter(entries, 103.1), entries[0])
        self.assertIs(next_waiter([], 103.1), None)

    def test_fifo_for_ties(self):
        a, b = [500, 1., 1, 200], [500, 1., 2, 200]
        self.assertIs(next_waiter([b, a], 1.5), a)

if __name__ == '__main__':
    unittest.main()
