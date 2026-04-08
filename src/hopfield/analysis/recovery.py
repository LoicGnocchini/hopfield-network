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
random_patterns = network.generate_patterns(8,30, rng=rng)
W_hebb = hebb.weight_hebb(random_patterns)


def compute_recovery_ratio():

    return