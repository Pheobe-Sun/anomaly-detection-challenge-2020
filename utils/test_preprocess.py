import unittest
import numpy as np
from preprocess import *  

class TestPreprocess(unittest.TestCase):

    def test_window_offset(self):
        expected = np.array([   
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  2],
        [ 0,  1,  2,  3],
        [ 1,  2,  3,  4],
        [ 2,  3,  4,  5],
        [ 3,  4,  5,  6],
        [ 4,  5,  6,  7],
        [ 5,  6,  7,  8],
        ])
        test_x = np.array([1,2,3,4,5,6,7,8,9])
        actual = window_offset(np.array([1,2,3,4,5,6,7,8,9]), 4) 
        num_incorrect= np.sum(1 - np.equal(actual,expected))
        self.assertTrue(num_incorrect == 0)
    
    def test_window(self):
        expected = np.array([   
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  2],
        [ 0,  1,  2,  3],
        [ 1,  2,  3,  4],
        [ 2,  3,  4,  5],
        [ 3,  4,  5,  6],
        [ 4,  5,  6,  7],
        [ 5,  6,  7,  8],
        [ 6,  7,  8,  9]
        ])
        test_x = np.array([1,2,3,4,5,6,7,8,9])
        actual = window(np.array([1,2,3,4,5,6,7,8,9]), 4) 
        num_incorrect= np.sum(1 - np.equal(actual,expected))
        self.assertTrue(num_incorrect == 0)
    
if __name__ == '__main__':
    unittest.main()