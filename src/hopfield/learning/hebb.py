import numpy as np

from numba import njit
from timer_wrapper import timer
from numpy.typing import NDArray

"""
Hebb training for weight matrix
"""

def weight_hebb(P: NDArray[np.int8]) -> NDArray[np.float64]:

    nombre_patterns, N = P.shape

    W = (np.sum(np.array([np.outer(p,p) for p in P]) - np.identity(N), axis=0))/ N

    return W



if __name__ == "__main__":

    P_test = np.array([[1,-1,-1,1,1],[-1,1,-1,1,1],[1,1,1,-1,-1]])

    print(weight_hebb(P_test))