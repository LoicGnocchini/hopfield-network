# Hopfield Network

A Python implementation of a Hopfield network that I used to explore associative memory and pattern retrieval.

The project includes Hebbian and perceptron-based learning rules, synchronous and asynchronous updates, and tools to study how well stored patterns can be recovered after being corrupted.

## What I explored

* Storage and retrieval of binary patterns
* Hebbian learning
* Perceptron-based learning
* Synchronous and asynchronous updates
* Pattern corruption and reconstruction
* Overlap between retrieved and stored patterns
* Energy evolution during retrieval

Some of the experiments use randomly generated patterns, while others use binary images.

## Project structure

* `src/hopfield/` — main implementation
* `experiments/` — experiment scripts
* `notebooks/` — exploratory notebooks
* `figures/` — figures generated from the experiments
* `results/` — saved experimental results

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/LoicGnocchini/hopfield-network.git
cd hopfield-network
pip install -r requirements.txt
```

The project requires Python 3.9 or newer.
