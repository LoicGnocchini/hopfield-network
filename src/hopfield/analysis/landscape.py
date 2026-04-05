import numpy as np
from numba import njit
from sklearn.decomposition import  PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image

from hopfield.analysis.energy import compute_energy
from hopfield import network

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = (7, 5)

celeba = np.load("src/hopfield/data/celeba_800_100x100_pm1_flat.npy")

patterns = np.array([celeba[4],  celeba[5],  celeba[6],  celeba[7], 
                     celeba[11], celeba[13], celeba[14], celeba[19]])