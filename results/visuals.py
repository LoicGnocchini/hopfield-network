import numpy as np
import matplotlib.pyplot as plt
from hopfield.analysis import recovery, hamming, overlap
from hopfield.learning import hebb, perceptron
from hopfield.utils import corrupt
from hopfield import network
from numpy.typing import NDArray
import matplotlib.colors as mcolors
from PIL import Image


plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = (7, 5)

celeba = np.load("src/hopfield/data/celeba_800_100x100_pm1_flat.npy")
fashionMNIST = np.load("src/hopfield/data/fashionMNIST_15_28x28_pm1_flat.npy")

patterns_mnist = fashionMNIST[[0, 8, 23]]

def plot_patterns(patterns: NDArray[np.int64],
                  noise_amount: float,
                  ) -> None:
    

    celeba_patterns = celeba[[4, 5, 6, 7]]
    weight_celeba_hebb = hebb.weight_hebb(celeba_patterns)
    # weight_celeba_perceptron = perceptron.weight_perceptron(celeba_patterns)

    corrupted_celeba = np.array([corrupt.corrupt_pattern(p, noise_amount, network.rng) for p in celeba_patterns])

    post_hebb = np.array([network.run_network(weight_celeba_hebb, c, network.rng)[0] for c in corrupted_celeba])
    # post_perceptron = np.array([network.run_network(weight_celeba_perceptron, c, network.rng)[0] for c in corrupted_celeba])

    plot_array_hebb = np.append(celeba_patterns, np.append(corrupted_celeba, post_hebb, axis=0), axis=0)
    # plot_array_perceptron = np.append(corrupted_celeba, post_perceptron, axis=0)

    hamming_dist = np.array([hamming.compute_hamming(post_hebb[i], celeba_patterns[i]) for i in range(post_hebb.shape[0])])

    cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(10, 8))
    for i in range(plot_array_hebb.shape[0]):
        plt.subplot(3,4,i+1)
        plt.imshow(plot_array_hebb[i].reshape(100,100), cmap=cmap_nb, norm=norm)
        if i < 4:
            plt.title("Original")
        elif i >= 8:
            plt.title(f"Hamming: {hamming_dist[i % 4]}")
            
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.suptitle(f"Pattern recognition with Hebbian learning at {noise_amount * 100:.0f}% noise, P={celeba_patterns.shape[0]}", fontsize=16)
    plt.savefig(f"figures/celeba_samples_noise{noise_amount}.pdf")
    plt.show()


def animate_step_by_step(state_orig: NDArray[np.int64],
                         target_pattern: NDArray[np.int64], 
                         W: NDArray[np.float64], 
                         image_shape: tuple[int, int],
                         rng: np.random.Generator,
                         max_sweeps: int = 5,
                         pause_time: float = 0.0001) -> NDArray[np.int64]:
    """
    Anime la récupération asynchrone mise à jour par mise à jour (pixel par pixel).
    """
    state = state_orig.copy().astype(np.float64)
    N = state.size
    side_x, side_y = image_shape
    
    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    # Affichage initial
    img_display = ax.imshow(state.reshape(image_shape), cmap=cmap_nb, norm=norm)
    
    # Un petit point rouge pour montrer quel neurone est évalué
    current_pixel, = ax.plot([], [], 'ro', markersize=15) 
    ax.axis('off')
    
    for sweep in range(max_sweeps):
        changed = False
        indexes = rng.integers(0, N, size=N)
        
        for step, index in enumerate(indexes):
            # Calcul du champ local (h)
            h = np.dot(W[index, :], state)
            new_val = 1.0 if h > 0 else -1.0
            
            if new_val != state[index]:
                state[index] = new_val
                changed = True
            
            # MISE À JOUR VISUELLE À CHAQUE NEURONE
            img_display.set_data(state.reshape(image_shape))
            
            # Calculer les coordonnées (x, y) du neurone pour placer le point rouge
            y_coord = index // side_x
            x_coord = index % side_x
            current_pixel.set_data([x_coord], [y_coord])
            
            ax.set_title(f"Sweep: {sweep + 1} | Évaluation du neurone: {step + 1}/{N}")
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(pause_time)

            current_overlap = np.abs(overlap.compute_overlap(state.astype(np.int8), target_pattern.astype(np.int8)))
            
            if current_overlap >= 0.99:
                current_pixel.set_data([], []) # Cacher le point rouge
                fig.canvas.draw()
                plt.ioff()
                plt.show()
                break
            
        if not changed:
            break
            
    current_pixel.set_data([], []) # Cacher le point rouge à la fin
    plt.ioff() 
    plt.show() 
    
    return state.astype(np.int64)

if __name__ == "__main__":

    # x -> 1, . --> -1
    to_pat = lambda s: np.array([-1 if c == 'X' else 1 for c in s if c in '.X'], dtype=np.int8)

    # 1. checker
    checker = np.fromfunction(lambda i, j: (i + j) % 2 * 2 - 1, (10, 10), dtype=np.int8).flatten()

    # 2. house
    house = to_pat("""
        ....XX....
        ...XXXX...
        ..XXXXXX..
        .XXXXXXXX.
        XXXXXXXXXX
        ..XXXXXX..
        ..XXXXXX..
        ..XX..XX..
        ..XX..XX..
        ..XX..XX..
    """)

    # 3. pi

    pi = to_pat("""
    .........X
    .XXXXXXXXX
    XXXXXXXXX.
    X.XX..XX..
    ..XX..XX..
    ..XX..XX..
    ..XX..XX..
    ..XX..XX..
    ..XX..XX..
    ..XX...XX.
    """)

    patts = np.array([checker, house, pi])

    # plot_patterns(celeba[[4, 5, 6, 7]], noise_amount=0.40)
    # plot_patterns(celeba[[4, 5, 6, 7]], noise_amount=0.45)
    # plot_patterns(celeba[[4, 5, 6, 7]], noise_amount=0.50)

    animate_step_by_step(corrupt.corrupt_pattern(patts[2], 0.20, network.rng).astype(np.int64),
                         patts[2], hebb.weight_hebb(patts), (10, 10), network.rng)