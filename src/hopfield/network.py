import numpy as np
from numpy.typing import NDArray
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from numba import njit
from  timer_wrapper import timer

from hopfield.learning import hebb, perceptron
from hopfield.update import asynchronous
from hopfield.utils import corrupt
from hopfield.analysis import overlap


rng = np.random.default_rng(seed=0)

"""
hopfield network function
"""


def generate_patterns(num_patterns: int, 
                      m: int, 
                      rng: np.random.Generator
                      ) ->NDArray[np.int8]:
    """
    Random patterns
    """
    return rng.choice([-1,1], size=(num_patterns, m**2))


# @timer
def run_network(Patterns_arr: NDArray[np.int8], 
                Pattern_corrupt: NDArray[np.int8], 
                learn: Callable,
                rng: np.random.Generator,
                ) ->NDArray[np.int8]:
    
    state = Pattern_corrupt
    W = learn(Patterns_arr)

    return asynchronous.update_asynch(state, W, rng)
    
 



if __name__ == "__main__":

     # template pour cmap noir et blanc-----------------------------------------
    cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    

    patterns = generate_patterns(10, 10, rng)
    # corr_patterns = np.array([p for p in corrupt.corrupt_pattern(patterns, 0.10, rng)])
    corr_pattern = corrupt.corrupt_pattern(patterns[0], 0.4, rng)

    cleaned_pattern = run_network(patterns, corr_pattern, 
                                  hebb.weight_hebb, rng)

    
# -----------------------------random patterns-----------------------------------
    corr_pat_matrix = corr_pattern.reshape(10,10)
    pattern_0_matrix = patterns[0].reshape(10,10)
    cleaned_pattern_matrix = cleaned_pattern.reshape(10,10)

    recouvrements = []
    for pattern in patterns:
        recouvrements.append(overlap.compute_overlap(cleaned_pattern, pattern))
    
    print(recouvrements)
   

    plt.subplot(1,3,1)
    plt.imshow(pattern_0_matrix, cmap=cmap_nb, norm=norm)
    plt.title("initial pattern")

    plt.subplot(1,3,2)
    plt.imshow(corr_pat_matrix, cmap=cmap_nb, norm=norm)
    plt.title("corrupted pattern")

    plt.subplot(1,3,3)
    plt.imshow(cleaned_pattern_matrix, cmap=cmap_nb, norm=norm)
    plt.title("cleaned pattern")

    plt.show() 


#--------------------------------------------------------------------------------
#--------------------------------XAV & LOUNA-------------------------------------

    louna = np.array(np.loadtxt("src/hopfield/data/image1_100x100_matrix.txt"),
                      dtype= np.int8)
    xavier = np.array(np.loadtxt("src/hopfield/data/image2_100x100_matrix.txt"), 
                      dtype=np.int8)
    xavier2 = np.array(np.loadtxt("src/hopfield/data/image3_100x100_matrix.txt"), 
                      dtype=np.int8)

    patterns_xl = np.array([  louna.reshape(10_000,),
                             xavier.reshape(10_000,), 
                            xavier2.reshape(10_000,)])

    # -------------corrupt---------------
    # louna_corr   = corrupt.corrupt_pattern(  louna.reshape(10_000,), 0.4, rng)
    # xavier_corr  = corrupt.corrupt_pattern( xavier.reshape(10_000,), 0.4, rng)
    # xavier2_corr = corrupt.corrupt_pattern(xavier2.reshape(10_000,), 0.4, rng)


    # ---------focused_corrupt-----------
    # louna_corr_2   = corrupt.corrupt_focused_pattern(  louna.reshape(10_000,), 0.40, rng)
    # xavier_corr_2  = corrupt.corrupt_focused_pattern(  xavier.reshape(10_000,), 0.40, rng)
    # xavier2_corr_2 = corrupt.corrupt_focused_pattern(  xavier2.reshape(10_000,), 0.40, rng)
    


    # ---------------cleaned hebb -----------------
    # louna_clean   = run_network(patterns_xl, louna_corr,
    #                             hebb.weight_hebb, rng)
    # xavier_clean  = run_network(patterns_xl, xavier_corr,
    #                             hebb.weight_hebb, rng)
    # xavier2_clean = run_network(patterns_xl, xavier2_corr, 
    #                             hebb.weight_hebb, rng)

    # -------------cleaned perceptron--------------
    # louna_clean   = run_network(patterns_xl, louna_corr,
    #                             perceptron.weight_perceptron, rng)
    # xavier_clean  = run_network(patterns_xl, xavier_corr,
    #                             perceptron.weight_perceptron, rng)
    # xavier2_clean = run_network(patterns_xl, xavier2_corr, 
    #                             perceptron.weight_perceptron, rng)


    # -------------corrupt plots-----------------
    # plt.subplot(2,3,1)
    # plt.imshow(louna_corr.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("louna corrupted")

    # plt.subplot(2,3,4)
    # plt.imshow(louna_clean.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("louna cleaned")

    # plt.subplot(2,3,2)
    # plt.imshow(xavier_corr.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("xavier corrupted")

    # plt.subplot(2,3,5)
    # plt.imshow(xavier_clean.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("xavier cleaned")

    # plt.subplot(2,3,3)
    # plt.imshow(xavier2_corr.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("xavier2 corrupted")

    # plt.subplot(2,3,6)
    # plt.imshow(xavier2_clean.reshape(100,100), cmap=cmap_nb, norm=norm)
    # plt.title("xavier2 cleaned")

    plt.show()

    # ---------focused corrupt plots----------
    # plt.subplot(2,1,1)
    # plt.imshow(louna_corr_2.reshape(100,100), cmap=cmap_nb, norm=norm)

    # plt.subplot(2,1,2)
    # plt.imshow(louna_clean.reshape(100,100), cmap=cmap_nb, norm=norm)

    # plt.show()
