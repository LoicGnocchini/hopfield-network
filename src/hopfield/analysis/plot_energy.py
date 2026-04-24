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
    
    state, energy_arr = asynchronous.update_asynch(state, weight_matrix, network.rng)

    plt.plot(energy_arr)
    plt.xlabel('sweep')
    plt.ylabel('energy')
    plt.title('Energy vs. Sweep')
    plt.show()


if __name__ == "__main__":

    patterns = network.generate_patterns(5, 10, network.rng)
    W = hebb.weight_hebb(patterns)

    init_state = corrupt.corrupt_pattern(patterns[0], 0.3, network.rng)

    plot_energy(init_state.astype(np.int64), W)