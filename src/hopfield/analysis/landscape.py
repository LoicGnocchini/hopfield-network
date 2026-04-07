import numpy as np
from numba import njit
from sklearn.decomposition import  PCA

from numpy.typing import NDArray
from typing import Callable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image

from hopfield.analysis.energy import compute_energy
from hopfield import network
from hopfield.learning import hebb, perceptron

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = (7, 5)

celeba = np.load("src/hopfield/data/celeba_800_100x100_pm1_flat.npy")

patterns = np.array([celeba[4],  celeba[5],  celeba[6],  celeba[7], 
                     celeba[11], celeba[13], celeba[14], celeba[19]])

W = hebb.weight_hebb(patterns)

# def energy_patterns(patterns: NDArray[np.int8],
#                     learn: Callable
#                     )-> NDArray[np.float64]:
    
#     W = learn(patterns)
#     E_arr = np.empty(patterns.size)
#     for i,pattern in enumerate(patterns):
#         E_arr[i] = compute_energy(pattern, W)

#     return np.astype(E_arr, dtype=(np.float64))


def get_landscape_surface(patterns: NDArray[np.float64], 
                          W: NDArray[np.float64], 
                          resolution: int=50
                          ) -> tuple[NDArray, NDArray, NDArray]:
    pca = PCA(n_components=2).fit(patterns)
    
    x_range = np.linspace(-2, 2, resolution)
    y_range = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            state_nd = pca.inverse_transform([[X[i, j], Y[i, j]]])[0]
            state_bin = np.where(state_nd >= 0, 1.0, -1.0)
            Z[i, j] = compute_energy(state_bin, W)
            
    return X, Y, Z


def plot_energy_landscape(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)
    ax.plot_surface(X, Y, Z)
    plt.show()


def plot_energy_landscape_2(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    ax.plot_surface(X, Y, Z, cmap="jet", edgecolor="none", antialiased=True)
    ax.contour(X, Y, Z, levels=20, colors="k", linewidths=0.4)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Energy")
    ax.view_init(elev=45, azim=-45)
    ax.grid(False)

    plt.show()


def plot_energy_topology(X, Y, Z):
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, Z, levels=30, cmap="jet")
    plt.contour(X, Y, Z, levels=30, colors="black", linewidths=0.4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Energy")
    plt.show()


if __name__ == "__main__":
    X, Y, Z = get_landscape_surface(patterns, W)

    # plot_energy_landscape(X, Y, Z)
    # plot_energy_landscape_2(X, Y, Z)
    plot_energy_topology(X, Y, Z)
