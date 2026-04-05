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

celeba = np.load("src/hopfield/data/celeba_800_100x100_pm1_flat.npy")

@timer
def stability_deter(num_patterns: NDArray[np.int64],
                    learn: Callable,
                    rng: np.random.Generator = network.rng
                    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    stable_patterns = np.zeros(num_patterns.size, dtype=(np.int64))
    stable_percentile = np.zeros(num_patterns.size, dtype=(np.float64))

    for i,nP in enumerate(tqdm(num_patterns)):

        patterns = celeba[:nP]
        W = learn(patterns)

        for j,pattern in enumerate(patterns):
            state, _ = network.run_network(W, pattern, rng)
            
            if np.array_equal(state, pattern):
                stable_patterns[i] += 1

        stable_percentile[i] = stable_patterns[i] / nP

    return num_patterns, stable_percentile

@timer
def stability_rand(num_patterns: NDArray[np.int64],
                   learn: Callable,
                   rng: np.random.Generator = network.rng
                   ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    stable_patterns = np.zeros(num_patterns.size, dtype=(np.float64))
    stable_percentile = np.zeros(num_patterns.size, dtype=(np.float64))

    for i,nP in enumerate(tqdm(num_patterns)):
        patterns = network.generate_patterns(nP, 100, rng)
        W = learn(patterns)

        for j,pattern in enumerate(patterns):

            state, _ = network.run_network(W, pattern, rng)
            if np.array_equal(state,pattern):
                stable_patterns[i] += 1
        stable_percentile[i] = stable_patterns[i] / nP

    return num_patterns, stable_percentile


def plot_stability_deter():
    x, y = stability_deter(np.array([3, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32]),hebb.weight_hebb)

    idx_not_one = np.where(y != 1)[0][0]

    plt.plot(x, y * 100,'o', color='black')
    plt.axvline(float(x[idx_not_one - 1]), linestyle='--', color="red")
    plt.xlabel("number of patterns")
    plt.ylabel("Stable patterns (%)")
    plt.savefig("src/hopfield/figures/fig_2a.pdf")
    plt.show()


def plot_stability_rand():

    x, y = stability_rand(np.array([10, 25, 100, 200, 300, 400, 500, 600, 700, 800]), hebb.weight_hebb)

    idx_not_one = np.where(y != 1)[0][0]

    plt.plot(x, y * 100, 'o', color='black')
    plt.axvline(float(x[idx_not_one - 1]), linestyle='--', color="red")
    plt.xlabel("number of patterns")
    plt.ylabel("Stable patterns (%)")
    plt.savefig("src/hopfield/figures/fig_2b.pdf")
    plt.show()



if __name__ == "__main__":
    
    plot_stability_deter()
    # plot_stability_rand()