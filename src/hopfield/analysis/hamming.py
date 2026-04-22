import numpy as np
from numba import njit
from numpy.typing import NDArray


"""
hamming distance function, takes two args; network's state and original
pattern
"""

def compute_hamming(state: NDArray[np.int64], 
                    pattern: NDArray[np.int64]
                    ) -> float:
    
    return int((state.size - np.dot(pattern,state)) / 2)

    

def normalized_hamming(state: NDArray[np.int64], 
                      pattern: NDArray[np.int64]
                      ) -> float:
    
    return float((state.size - np.dot(pattern,state)) / (2*state.size))