import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from numba import njit
from numpy.typing import NDArray
from typing import Callable

from hopfield.analysis import hamming, overlap
from hopfield.learning import hebb, perceptron
from hopfield.utils import corrupt
from hopfield import network
from timer_wrapper import timer

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = (7, 5)

rng = network.rng
f_MNIST = np.load("src/hopfield/data/fashionMNIST_15_28x28_pm1_flat.npy")

def compute_recovery_ratio_deter(patterns: NDArray[np.int8],
                           weight: NDArray[np.float64],
                           noise_amounts: NDArray[np.float64],
                           rng: np.random.Generator
                           ) ->NDArray[np.float64]:
    
    N_success = np.zeros(noise_amounts.size)
    for i,noise in enumerate(tqdm(noise_amounts)):
        for pattern in patterns:
            corr_pattern = corrupt.corrupt_pattern(pattern, noise, rng)
            state, _ = network.run_network(weight, corr_pattern, rng)

            if overlap.compute_overlap(state, pattern) >= 0.95:
                N_success[i] += 1
    
    return N_success / patterns.shape[0]


if __name__ == "__main__":

    

    patterns_mnist = f_MNIST[[0, 8, 23]]

    W_hebb = hebb.weight_hebb(patterns_mnist)
    # W_perceptron = perceptron.weight_perceptron(patterns_mnist)

    noise_arr = np.linspace(0.38,0.50,50)

    ratio = compute_recovery_ratio_deter(patterns_mnist, W_hebb, noise_arr, rng)
    print(ratio)
    

    plt.plot(noise_arr, ratio)
    plt.show()