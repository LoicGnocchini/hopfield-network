import numpy as np
from numba import njit
from numpy.typing import NDArray

"""
recovery function, takes two arguments ; the network's state and an
array of all the memorised patterns. Returns an array of the recovery 
of each pattern.
"""

def calculate_recovery(state: NDArray[np.int8], 
                       patterns: NDArray[np.int8]
                       ) -> NDArray[np.float64]:

    return np.dot(patterns, state)/state.size