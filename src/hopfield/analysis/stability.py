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
mnist = np.load("src/hopfield/data/fashionMNIST_2000_784_pm1_flat.npy")

@timer
def stability_deter(num_patterns: NDArray[np.int64],
                    learn: Callable,
                    data_base: NDArray[np.float64],
                    rng: np.random.Generator = network.rng
                    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    stable_patterns = np.zeros(num_patterns.size, dtype=(np.int64))
    stable_percentile = np.zeros(num_patterns.size, dtype=(np.float64))

    for i,nP in enumerate(tqdm(num_patterns)):

        patterns = data_base[:nP]
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
                   data_base: NDArray[np.float64],
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


def plot_stability_deter(learn: Callable,
                         nb_patterns_arr: NDArray[np.int64],
                         data_base: NDArray[np.float64]
                         ) -> None:
    x, y = stability_deter(nb_patterns_arr,learn, data_base)

    idx_not_one = np.where(y != 1)[0]
    if len(idx_not_one) > 0:
        idx_not_one = idx_not_one[0]
        plt.axvline(float(x[idx_not_one - 1]), linestyle='--', color="red")

    plt.plot(x, y * 100,'o', color='black')
    plt.xlabel("number of patterns")
    plt.ylabel("Stable patterns (%)")
    plt.savefig(f"figures/fig_2{'a' if learn == hebb.weight_hebb else 'c'}.pdf")
    plt.show()


def plot_stability_rand(learn: Callable,
                        nb_patterns_arr: NDArray[np.int64],
                        data_base: NDArray[np.float64]
                        ) -> None:

    x, y = stability_rand(nb_patterns_arr, learn, data_base)

    idx_not_one = np.where(y != 1)[0]

    if len(idx_not_one) > 0:
        idx_not_one = idx_not_one[0]
        plt.axvline(float(x[idx_not_one - 1]), linestyle='--', color="red")


    plt.plot(x, y * 100, 'o', color='black')
    plt.xlabel("number of patterns")
    plt.ylabel("Stable patterns (%)")
    plt.savefig(f"figures/fig_2{'b' if learn == hebb.weight_hebb else 'd'}.pdf")
    plt.show()



if __name__ == "__main__":
    
    nb_patterns_hebb_1 = np.array([3, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    nb_patterns_hebb_2 = np.array([10, 25, 100, 200, 300, 400, 500, 600, 700, 800])

    nb_patterns_perc_1 = np.array([10,100,1000])
    nb_patterns_perc_2 = np.array([10, 25, 100, 200, 300, 400, 500, 600, 700, 800])

    # plot_stability_deter(hebb.weight_hebb, nb_patterns_hebb_1)
    # plot_stability_rand(hebb.weight_hebb, nb_patterns_hebb_2)
    plot_stability_deter(perceptron.weight_perceptron, nb_patterns_perc_1, mnist)
    # plot_stability_rand(perceptron.weight_perceptron, nb_patterns_perc_2, mnist)