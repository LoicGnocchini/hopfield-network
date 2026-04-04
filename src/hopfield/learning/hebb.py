import numpy as np

from numba import njit
from timer_wrapper import timer
from numpy.typing import NDArray

"""
Hebb training for weight matrix
"""
@njit
def weight_hebb(P: NDArray[np.int8]) -> NDArray[np.float64]:

    _, N = P.shape
    Pf = P.astype(np.float64)
    W = (Pf.T @ Pf)/ N
    np.fill_diagonal(W,0.0)

    return W



if __name__ == "__main__":

    P_test = np.array([[1,-1,-1,1,1],[-1,1,-1,1,1],[1,1,1,-1,-1]])

    print(weight_hebb(P_test))