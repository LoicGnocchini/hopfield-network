import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numba 

from numpy.typing import NDArray
from typing import Callable
from hopfield import utils
from hopfield import network
from hopfield.utils import corrupt
from hopfield.analysis import overlap, hamming
from hopfield.learning import hebb, perceptron

rng = network.rng

"""
Here we test the robustness of the patterns vs % of noise, with different amounts 
of patterns stored. We use the recovery function and consider m < 0.05 to be robust.
"""


# for num_patterns amount of random patterns
def compute_recovery_robustness(num_patterns: int, 
                            noise_amounts: NDArray[np.float64],
                            size: int,
                            learning: Callable
                            ) ->tuple[NDArray[np.float64],
                                      NDArray[np.float64],
                                      NDArray[np.float64]]:
    
    patterns = network.generate_patterns(num_patterns,size, rng)
    W = learning(patterns)

    mean_overlap_arr = []
    std_overlap_arr = []
    for n,noise in enumerate(noise_amounts):

        overlaps = []
        for i,pattern in enumerate(patterns):

            corr_pattern = corrupt.corrupt_pattern(pattern, noise, rng)
            recovered_pat, _ = network.run_network(W, corr_pattern, rng)

            overlaps.append(overlap.compute_overlap(recovered_pat, pattern))

        mean_overlap_arr.append(np.mean(overlaps))
        std_overlap_arr.append(np.std(overlaps))
        print(f"{num_patterns} patterns, noise: {n}/{noise_amounts.size}")

    mean_overlap_arr = np.asarray(mean_overlap_arr, dtype=np.float64)
    std_overlap_arr = np.asarray(std_overlap_arr,dtype= np.float64)

    return mean_overlap_arr, noise_amounts, std_overlap_arr


def compute_threshold() ->int:

    number_patterns = np.array([12,13,14,15,16,17])

    means_overlap = np.zeros(number_patterns.size)
    for i,P in enumerate(number_patterns):
        (mean_overlap, _ , _) = compute_recovery_robustness(P,np.zeros(1),10,hebb.weight_hebb)

        means_overlap[i] = mean_overlap

        if mean_overlap < 0.99:
            return number_patterns[i-1]
    else:
        raise ValueError("No threshold found")
        


def plot_overlap_vs_noise():
    noises_arr = np.linspace(0, 1, 50)
    number_pattern = [5, 10, 15, 25, 50]

    N = len(number_pattern)
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, N)]

    plt.figure(figsize=(18,5))
    for i, P in enumerate(number_pattern):
        overlaps, noises, error = compute_recovery_robustness(P, noises_arr, 10,
                                                            hebb.weight_hebb)
        plt.subplot(1,2,1)
        plt.plot(noises,overlaps, '-', color=colors[i], markersize=2, label=f"{P} patterns")
        plt.fill_between(noises, overlaps - error, overlaps + error, color=colors[i], alpha=0.1)

        plt.subplot(1,2,2)
        plt.plot(noises,overlaps, '-', color=colors[i], markersize=2, alpha=0.1)
        
    plt.subplot(1,2,1)
    plt.xlabel("Initial noise")
    plt.ylabel("Mean overlap")
    plt.legend()

    threshold = compute_threshold()
    threshold_overlaps, noises, treshold_err = compute_recovery_robustness(threshold, noises_arr, 10,
                                                            hebb.weight_hebb)
    plt.subplot(1,2,2)
    plt.plot(noises,threshold_overlaps, '-', color="green", markersize=2, label=f"{threshold} patterns")
    plt.fill_between(noises, threshold_overlaps - treshold_err, threshold_overlaps + treshold_err, color="green", alpha=0.1)
    plt.ylabel("Mean overlap")
    plt.xlabel("Initial noise")
    plt.legend()

    plt.show()


def main():
    plot_overlap_vs_noise()


if __name__ == "__main__":

    main()


    