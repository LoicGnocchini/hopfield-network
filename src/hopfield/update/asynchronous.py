import numpy as np
from numba import njit
from timer_wrapper import timer
from numpy.typing import NDArray

from hopfield.analysis.energy import compute_energy

"""
asyncrhonous update of neurons
"""

def rand_indexes(N: int,
                 sweeps: int, 
                 rng: np.random.Generator
                 ) -> NDArray[np.int64]:
    
    indexes = rng.integers(0, N, size=sweeps * N)
    return indexes

# @timer
@njit
def asynch(state_orig: NDArray[np.float64], 
           W: NDArray[np.float64],
           indexes: NDArray[np.int64]
           ) ->tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    asynchronous update; calulating the local field of neurons one by one 
    randomly until given number of sweeps or convergence.
    """
    state = state_orig.copy()
    sweeps = indexes.size // state.size
    energy_arr = np.empty(state.size * sweeps, dtype=(np.float64))
    energy_arr[0] = compute_energy(state, W)
    count = 1

    for sweep in range(sweeps):
        changed = False
        start = sweep * state.size
        end = (sweep + 1) * state.size

        for k in range(start, end):
            index = indexes[k]
            # h = np.dot(W[index,:],state)
            h = 0.0
            for j in range(state.size):
                h += W[index, j] * state[j]

            new_val = 1.0 if h > 0 else -1.0
            
            if new_val != state[index]:
                state[index] = new_val
                changed = True

        energy_arr[count] = compute_energy(state, W)
        count += 1

        if not changed:
            break

    return state.astype(np.int64), energy_arr[:count]


def update_asynch(state_orig: NDArray[np.int64], 
                  W: NDArray[np.float64],
                  rng: np.random.Generator,
                  ) ->tuple[NDArray[np.int64], NDArray[np.float64]]:

    sweeps = 1000
    indexes_arr = rand_indexes(state_orig.size, sweeps, rng)

    return asynch(state_orig.astype(np.float64), W, indexes_arr)
