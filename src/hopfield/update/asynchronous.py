import numpy as np

from numba import njit
from wrappers import timer
from numpy.typing import NDArray

"""
asyncrhonous update of neurons
"""
# @njit
@timer
def update_asynch(state_orig: NDArray[np.int8], W: NDArray[np.float64],
                  rng: np.random.Generator) ->NDArray[np.int8]:
    """
    asynchronous update; calulating the local field of neurons one by one 
    randomly until given number of sweeps or convergence.
    """
    sweeps = 1000
    state = state_orig.copy()

    for _ in range(sweeps):
        state_temp = state.copy()
        for _ in range(len(state)):

            i = rng.integers(0,len(state))
            h = np.dot(W[i,:],state)

            if h > 0:
                state[i] = 1    # s_i = +1
    
            elif h < 0:
                state[i] = -1   # s_i = -1
    
            else:
                pass            # s_i = s_i 

        if np.array_equal(state,state_temp):
            break

    return state