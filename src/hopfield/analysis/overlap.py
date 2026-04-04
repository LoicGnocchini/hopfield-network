import numpy as np
from numba import njit
from numpy.typing import NDArray

"""
overlap function, takes two arguments ; the network's state and the corresponding
pattern. Returns the overlap.
"""

def compute_overlap(state: NDArray[np.int8], 
                       pattern: NDArray[np.int8]
                       ) -> float:

    return float(np.dot(pattern, state)/state.size)



