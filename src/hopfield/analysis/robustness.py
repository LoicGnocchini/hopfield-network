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
            recovered_pat, _ = network.run_network(W, corr_pattern.astype(np.int64), rng)

            overlaps.append(overlap.compute_overlap(recovered_pat.astype(np.int8), pattern))

        mean_overlap_arr.append(np.mean(overlaps))
        std_overlap_arr.append(np.std(overlaps))
        print(f"{num_patterns} patterns, noise: {n}/{noise_amounts.size}")

    mean_overlap_arr = np.asarray(mean_overlap_arr, dtype=np.float64)
    std_overlap_arr = np.asarray(std_overlap_arr,dtype= np.float64)

    return mean_overlap_arr, noise_amounts, std_overlap_arr


def compute_threshold(learning_rule: Callable) ->int:

    if learning_rule == hebb.weight_hebb:
        number_patterns = np.array([12,13,14,15,16,17])

    else:
        number_patterns = np.array([25,26,27,28,29,30])

    means_overlap = np.zeros(number_patterns.size)
    for i,P in enumerate(number_patterns):
        (mean_overlap, _ , _) = compute_recovery_robustness(P,np.zeros(1),10,learning_rule)

        means_overlap[i] = mean_overlap[0]

        if mean_overlap < 0.99:
            return number_patterns[i-1]
    else:
        raise ValueError("No threshold found")
        


def plot_overlap_vs_noise():
    noises_arr = np.linspace(0, 1, 50)
    number_pattern_hebb = [5, 10, 15, 25, 50]
    number_pattern_perc = [5, 10, 15, 25 ,50]

    cmap_hebb = plt.get_cmap("plasma")
    colors_hebb = [cmap_hebb(i) for i in np.linspace(0, 1, len(number_pattern_hebb))]
    cmap_perc = plt.get_cmap("viridis")
    colors_perc = [cmap_perc(i) for i in np.linspace(0, 1, len(number_pattern_perc))]

# ----------------------------------HEBB------------------------------------------
    plt.figure(figsize=(18,5))
    plt.suptitle("Robustness of the patterns vs % of noise, with different amounts of patterns stored")
    for i, P in enumerate(number_pattern_hebb):
        overlaps_hebb, noises_hebb, error_hebb = compute_recovery_robustness(P, noises_arr, 10,
                                                            hebb.weight_hebb)
        plt.subplot(1,2,1)
        plt.plot(noises_hebb,overlaps_hebb, '-', color=colors_hebb[i], markersize=2, label=f"{P} patterns")
        plt.fill_between(noises_hebb, overlaps_hebb - error_hebb, overlaps_hebb + error_hebb, color=colors_hebb[i], alpha=0.1)

        plt.subplot(1,2,2)
        plt.plot(noises_hebb,overlaps_hebb, '-', color=colors_hebb[i], markersize=2, alpha=0.1)
        
    plt.subplot(1,2,1)
    plt.xlabel("Initial noise")
    plt.ylabel("Mean overlap")
    plt.title("Hebbian learning")
    plt.legend()

    threshold = compute_threshold(hebb.weight_hebb)
    threshold_overlaps, noises, treshold_err = compute_recovery_robustness(threshold, noises_arr, 10,
                                                            hebb.weight_hebb)
    plt.subplot(1,2,2)
    plt.plot(noises,threshold_overlaps, '-', color="green", markersize=2, label=f"{threshold} patterns")
    plt.fill_between(noises, threshold_overlaps - treshold_err, threshold_overlaps + treshold_err, color="green", alpha=0.1)
    plt.ylabel("Mean overlap")
    plt.xlabel("Initial noise")
    plt.title("Hebbian threshold")
    plt.legend()
    plt.savefig("figures/fig_1a.pdf")
    plt.show()

#------------------------------PERCEPTRON vs HEBB---------------------------------
    plt.figure(figsize=(18,5))
    plt.suptitle("Robustness of the patterns vs % of noise, with different amounts of patterns stored")
    for i, P in enumerate(number_pattern_perc):
        overlaps_perc, noises_perc, error_per = compute_recovery_robustness(P, noises_arr, 10,
                                                            perceptron.weight_perceptron)
        plt.subplot(1,2,1)
        plt.plot(noises_perc,overlaps_perc, '-', color=colors_perc[i], markersize=2, label=f"{P} patterns")
        plt.fill_between(noises_perc, overlaps_perc - error_per, overlaps_perc + error_per, color=colors_perc[i], alpha=0.1)

        overlaps_hebb, noises_hebb, error_hebb = compute_recovery_robustness(P, noises_arr, 10,
                                                            hebb.weight_hebb)
        plt.subplot(1,2,2)
        plt.plot(noises_hebb,overlaps_hebb, '-', color=colors_hebb[i], markersize=2, label=f"{P} patterns")
        plt.fill_between(noises_hebb, overlaps_hebb - error_hebb, overlaps_hebb + error_hebb, color=colors_hebb[i], alpha=0.1)
        
    plt.subplot(1,2,2)
    plt.xlabel("Initial noise")
    plt.ylabel("Mean overlap")
    plt.title("Hebbian learning")
    plt.legend()

    plt.subplot(1,2,1)
    plt.xlabel("Initial noise")
    plt.ylabel("Mean overlap")
    plt.title("Perceptron learning")
    plt.legend()

    plt.savefig("figures/fig_1b.pdf")
    plt.show()



def main():
    plot_overlap_vs_noise()


if __name__ == "__main__":

    main()


    