from lazer_final import Lazor_Solution

import unittest
from unittest.mock import MagicMock

class TestLazorSolution(unittest.TestCase):
    

    def setUp(self):
        # Mock input data
        self.input_data = {
            "o_l": [[0, 0], [1, 1], [2, 2]],
            "Size": [3, 3],
            "Lazers": [[0, 0, 1, 1]],
            "Points": [[2, 2]],
            "A": 1,
            "B": 1,
            "C": 1,
            "A_l": [],
            "B_l": [],
            "C_l": []
        }
        self.solution = Lazor_Solution(self.input_data)

    def test_initialization(self):
        # Test if Lazor_Solution initializes with correct values
        self.assertEqual(self.solution.o_l, [[0, 0], [1, 1], [2, 2]])
        self.assertEqual(self.solution.size, [3, 3])
        self.assertEqual(self.solution.lazers, [[0, 0, 1, 1]])
        self.assertEqual(self.solution.points, [[2, 2]])

    def test_pos_chk_within_bounds(self):
        # Test pos_chk for position within bounds
        self.assertTrue(self.solution.pos_chk((2, 2)))
        self.assertTrue(self.solution.pos_chk((0, 0)))

    def test_pos_chk_outside_bounds(self):
        # Test pos_chk for position outside bounds
        self.assertFalse(self.solution.pos_chk((3, 3)))
        self.assertFalse(self.solution.pos_chk((-1, -1)))

  

if __name__ == "__main__":
    unittest.main()
