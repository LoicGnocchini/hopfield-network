import numpy as np

from numba import njit
from numpy.typing import NDArray
from timer_wrapper import timer

from hopfield.learning.hebb import weight_hebb


"""
Perceptron training for weight matrix
"""
@timer
@njit
def weight_perceptron(P: NDArray[np.int8]) -> NDArray[np.float64]:

    number_patterns, N = P.shape
    Pf = np.ascontiguousarray(P.astype(np.float64))
    W = weight_hebb(P)
    
    eta = 0.1
    n_iter = 0

    while True:
        updated = False
        for pattern in Pf:

            for i in range(len(pattern)):
                
                W[i,i] = 0
               
                h = 0.0
                for j in range(N):
                    h += W[i, j] * pattern[j]

                if pattern[i] * h <= 0:
                    W[i,:] = W[i,:] + eta * ((pattern[i] * pattern))/ N 
                    W[:,i] = W[i,:] 
                    W[i,i] =0
                    updated = True
        n_iter += 1
        if (updated == False) or (n_iter == 100_000):
            print(n_iter)
            break 

    return W


if __name__ == "__main__":

    P_test = np.array([[1,-1,-1,1,1],[-1,1,-1,1,1],[1,1,1,-1,-1]])

    print(weight_perceptron(P_test))
    print(weight_hebb(P_test))