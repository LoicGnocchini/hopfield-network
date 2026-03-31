import numpy as np
from numba import njit
from numpy.typing import NDArray


"""
hamming distance function
"""

def calculate_hamming(state: NDArray[np.int8], 
                      pattern: NDArray[np.int8]
                      ) -> float:
    
    return int((state.size - np.dot(pattern,state)) / 2)

    

def normalized_hamming(state: NDArray[np.int8], 
                      pattern: NDArray[np.int8]
                      ) -> float:
    
    return float((state.size - np.dot(pattern,state)) / (2*state.size))