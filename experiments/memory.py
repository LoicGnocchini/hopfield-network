import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

celeba = np.load("src/hopfield/data/celeba_800_100x100_pm1_flat.npy")

cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
norm = mcolors.Normalize(vmin=-1, vmax=1)



plt.imshow(celeba[26].reshape(100,100), cmap=cmap_nb, norm=norm)
plt.show()


# celeba[4], celeba[5], celeba[6], celeba[7], celeba[11], celeba[13], celeba[14], celeba[19] : [4, 5, 6, 7, 11, 13, 14, 19]
# celeba[26] 