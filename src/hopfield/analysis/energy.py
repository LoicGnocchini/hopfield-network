import numpy as np
from numba import njit
from numpy.typing import NDArray

"""
energy function
"""

@njit
def compute_energy(state: NDArray[np.float64], 
                   weight_matrix: NDArray[np.float64]
                   ) -> float:

    return float(- 0.5 * state @ weight_matrix @ state)

    