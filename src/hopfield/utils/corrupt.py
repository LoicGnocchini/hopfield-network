import numpy as np

from numpy.typing import NDArray

def corrupt_pattern(P: NDArray[np.int8], percent: float, rng: np.random.Generator
                    ) -> NDArray[np.int8]:
    P_corrupt = P.copy()
    
    k = int(np.round(percent * len(P)))

    indexes = rng.choice(len(P),size=(k), replace=False)

    for i in indexes:
        P_corrupt[i] *= -1

    return P_corrupt


def corrupt_focused_pattern(P: NDArray[np.int8], percent: float, rng: np.random.Generator
                    ) -> NDArray[np.int8]:
    P_corrupt = P.copy()

    k = int(np.round(percent * len(P)))

    light = np.ones(k)

    P_corrupt[-k :] = light

    return P_corrupt
