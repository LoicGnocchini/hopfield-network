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

def compute_all_R(patterns: NDArray[np.int8],
                       weight: NDArray[np.float64],
                       noise_amounts: NDArray[np.float64],
                       rng: np.random.Generator,
                       n_tests: int = 50
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    
    R_correct = np.zeros(noise_amounts.size)
    R_parasite = np.zeros(noise_amounts.size)
    R_autre = np.zeros(noise_amounts.size) # Nouvelle catégorie
    total_essais = patterns.shape[0] * n_tests
    
    for i, noise in enumerate(tqdm(noise_amounts)):
        for target_idx, target_pattern in enumerate(patterns):
            for _ in range(n_tests):
                corr_pattern = corrupt.corrupt_pattern(target_pattern, noise, rng)
                state, _ = network.run_network(weight, corr_pattern, rng)

                overlaps = np.array([overlap.compute_overlap(state, p) for p in patterns])
                
                if overlaps[target_idx] >= 0.95:
                    R_correct[i] += 1
                else:
                    m_max = np.max(np.abs(overlaps))
                    if m_max < 0.95:
                        R_parasite[i] += 1
                    else:
                        # Converge vers un autre souvenir ou un inverse
                        R_autre[i] += 1 
                        
    return R_correct / total_essais, R_parasite / total_essais, R_autre / total_essais


if __name__ == "__main__":
    patterns_mnist = f_MNIST[[0, 8, 23]]
    W_hebb = hebb.weight_hebb(patterns_mnist)
    noise_arr = np.linspace(0.2, 0.8, 50)

    R_correct, R_parasite, R_autre = compute_all_R(patterns_mnist, W_hebb, noise_arr, rng, n_tests=50)
    
    plt.plot(noise_arr, R_correct, marker='o', markersize=5, color='blue', label="Récupération correcte ($R$)")
    plt.plot(noise_arr, R_parasite, marker='s', markersize=5, color='red', label="États parasites ($R_{parasite}$)")
    plt.plot(noise_arr, R_autre, marker='^', markersize=5, color='green', label="Autres états ($R_{autre}$)")

    plt.xlabel(r"Niveau de bruit initial ($\eta$)")
    plt.ylabel("Taux")
    plt.title("Bassin d'attraction et États Parasites (P=3, N=784)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.savefig("figures/fig_3.pdf")
    plt.show()
