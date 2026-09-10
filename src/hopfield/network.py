import numpy as np
from numpy.typing import NDArray
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from numba import njit
from  timer_wrapper import timer

from hopfield.learning import hebb, perceptron
from hopfield.update import asynchronous, synchronous
from hopfield.utils import corrupt
from hopfield.analysis import overlap


rng = np.random.default_rng(seed=0)

"""
hopfield network function
"""

# @timer
def generate_patterns(num_patterns: int, 
                      m: int, 
                      rng: np.random.Generator
                      ) ->NDArray[np.int8]:
    """
    Random patterns
    """
    return rng.choice([-1,1], size=(num_patterns, m**2))


# @timer
def run_network(weight: NDArray[np.float64], 
                Pattern_corrupt: NDArray[np.int64], 
                rng: np.random.Generator
                ) ->tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    state = Pattern_corrupt
    W = weight

    return asynchronous.update_asynch(state, W, rng)
    
def run_network_synchronous(weight: NDArray[np.float64], 
                            Pattern_corrupt: NDArray[np.int64], 
                            rng: np.random.Generator
                            ) ->tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    state = Pattern_corrupt
    W = weight

    return synchronous.update_synchronous(state, W)
 



if __name__ == "__main__":

     # template pour cmap noir et blanc-----------------------------------------
    cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
# ------------------------Test with random patterns-------------------------------

    patterns = generate_patterns(3, 20, rng)
    corr_pattern = corrupt.corrupt_pattern(patterns[0], 0.42, rng)

    recovered_pattern, energy_rdm_pat = run_network(hebb.weight_hebb(patterns), corr_pattern.astype(dtype=np.int64), rng)

    corr_pat_matrix = corr_pattern.reshape(20,20)
    pattern_0_matrix = patterns[0].reshape(20,20)
    recovered_pattern_matrix = recovered_pattern.reshape(20,20)

    overlap_tab = []
    for p in [patterns[0], corr_pattern, recovered_pattern]:
        overlap_tab.append(overlap.compute_overlap(p.astype(np.int8), patterns[0].astype(np.int8)))
    
    print("overlap with original pattern:", overlap_tab[0])
   
    plt.subplot(1,3,1)
    plt.imshow(pattern_0_matrix, cmap=cmap_nb, norm=norm)
    plt.title(f"initial pattern\noverlap = {overlap_tab[0]:.2f}")

    plt.subplot(1,3,2)
    plt.imshow(corr_pat_matrix, cmap=cmap_nb, norm=norm)
    plt.title(f"corrupted pattern\noverlap = {overlap_tab[1]:.2f}")

    plt.subplot(1,3,3)
    plt.imshow(recovered_pattern_matrix, cmap=cmap_nb, norm=norm)
    plt.title(f"recovered pattern\noverlap = {overlap_tab[2]:.2f}")

    plt.show() 
