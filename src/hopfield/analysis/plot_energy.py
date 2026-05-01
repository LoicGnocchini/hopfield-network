import numpy as np
from numpy.typing import NDArray


from hopfield import network
from hopfield.update import asynchronous, synchronous
from hopfield.learning import hebb, perceptron
from hopfield.utils import corrupt
import matplotlib.pyplot as plt



def plot_energy(state: NDArray[np.int64],
                weight_matrix: NDArray[np.float64],
                ) -> None:
    
    _, energy_arr = asynchronous.update_asynch(state, weight_matrix, network.rng)
    _, energy_arr_sync = synchronous.update_synchronous(state, weight_matrix)

    plt.plot(energy_arr, label='Energy, asynchronous update', color='blue')
    plt.plot(energy_arr_sync, label='Energy, synchronous update', color='orange')
    plt.xlabel('sweep')
    plt.ylabel('energy')
    plt.title('Energy vs. Sweep for Hebbian Learning Rule')
    plt.savefig("figures/energy_hebb.pdf")
    plt.legend()
    plt.show()


def plot_energy_perceptron(N_patterns: int,
                           ) -> None:
    
    patterns = network.generate_patterns(N_patterns, 100, network.rng)
    weight_matrix_perc = perceptron.weight_perceptron(patterns)
    corrupted_pattern = [corrupt.corrupt_pattern(pattern, 0.40, network.rng).astype(np.int64) for pattern in patterns]

    # Is the weight matrix symmetric or not? 
    W = weight_matrix_perc
    W_sym = 0.5 * (W + W.T)
    W_asym = 0.5 * (W - W.T)

    print("norm W:", np.linalg.norm(W))
    print("norm symmetric part:", np.linalg.norm(W_sym))
    print("norm antisymmetric part:", np.linalg.norm(W_asym))
    print("relative asymmetry:", np.linalg.norm(W_asym) / np.linalg.norm(W))
    print("max |W - W.T|:", np.max(np.abs(W - W.T)))


    for corrupted in corrupted_pattern:
        _, energy_arr_perc = asynchronous.update_asynch(corrupted, weight_matrix_perc, network.rng)

        plt.plot(energy_arr_perc, color='blue', alpha=0.3)
    plt.xlabel('sweep')
    plt.ylabel('energy')
    plt.title('Energy vs. Sweep for Perceptron Learning Rule')
    plt.savefig("figures/energy_perceptron.pdf")
    plt.show()


def plot_multiple_N_patterns(N_patterns: NDArray[np.int64],
                             ) -> None:
    E_hebb = []
    E_perc = []
    for i, n in enumerate(N_patterns):
        patterns = network.generate_patterns(n, 20, network.rng)
        weight_matrix_perc = perceptron.weight_perceptron(patterns)
        weight_matrix_hebb = hebb.weight_hebb(patterns)
        corrupted_pattern = [corrupt.corrupt_pattern(pattern, 0.3, network.rng).astype(np.int64) for pattern in patterns]

        # Is the weight matrix symmetric or not? 
        W = weight_matrix_perc
        W_sym = 0.5 * (W + W.T)
        W_asym = 0.5 * (W - W.T)

        print("norm W:", np.linalg.norm(W))
        print("norm symmetric part:", np.linalg.norm(W_sym))
        print("norm antisymmetric part:", np.linalg.norm(W_asym))
        print("relative asymmetry:", np.linalg.norm(W_asym) / np.linalg.norm(W))
        print("max |W - W.T|:", np.max(np.abs(W - W.T)))

        for corrupted in corrupted_pattern:
            _, energy_arr_hebb = asynchronous.update_asynch(corrupted, weight_matrix_hebb, network.rng)
            _, energy_arr_perc = asynchronous.update_asynch(corrupted, weight_matrix_perc, network.rng)

            plt.plot(energy_arr_hebb,'--', color='blue', alpha=0.5)
            plt.plot(energy_arr_perc, color='orange', alpha=0.5)

            E_hebb.append(energy_arr_hebb[-1])
            E_perc.append(energy_arr_perc[-1])


        if i == 0:
            plt.plot([], [], '--', color='blue', label='Hebbian learning')
            plt.plot([], [], color='orange', label='Perceptron learning')
            plt.legend()

    plt.xlabel('sweep')
    plt.ylabel('energy')
    plt.title('Energy vs. Sweep for Perceptron Learning Rule')
    plt.savefig("figures/energy_comparison.pdf")
    plt.show()

    print("Final energy for Hebbian learning:", np.mean(E_hebb))
    print("Final energy for Perceptron learning:", np.mean(E_perc))

if __name__ == "__main__":

    patterns = network.generate_patterns(20, 100, network.rng)
    W_hebb = hebb.weight_hebb(patterns)
    W_perceptron = perceptron.weight_perceptron(patterns)

    init_state = corrupt.corrupt_pattern(patterns[0], 0.3, network.rng)

    plot_energy(init_state.astype(np.int64), W_hebb)

    plot_multiple_N_patterns(np.array([5,20,50]))
    # plot_energy_perceptron(500)