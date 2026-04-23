import numpy as np
from numba import njit
from timer_wrapper import timer
from numpy.typing import NDArray

from hopfield.analysis.energy import compute_energy

def synch(state: NDArray[np.int64],
          W: NDArray[np.float64]
          ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    synchronous update; calculating the local field of all neurons at once and 
    updating them together until convergence or given number of sweeps.
    """
    energy_arr = []
    while True:
        energy_arr.append(compute_energy(state.astype(np.float64), W))
        h = np.dot(W, state.astype(np.float64))
        new_state = np.where(h > 0, 1, -1)

        if np.array_equal(new_state, state):
            break

        state = new_state

    return state.astype(np.int64), np.array(energy_arr, dtype=np.float64)