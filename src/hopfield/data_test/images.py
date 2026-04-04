from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

"""
Passer d'une image noir et blanc à une matrice {-1,1} 
"""


img1 = Image.open("docs/images/dither_it_projet3075_louna.jpg").convert("L")   
img2 = Image.open("docs/images/dither_it_projet3075_xav_2.jpg").convert("L")  
img3 = Image.open("docs/images/dither_it_projet3075_xav_gros.jpg").convert("L")   

counter = 0

def image2Matrix(img, format):
    global counter 
    counter += 1

    arr = np.array(img)

    mat = (arr > 128).astype(int) 

    mat_hopfield = 2 * mat - 1

    # np.savetxt(f"src/hopfield/data/images/image{counter}_{format}x{format}_matrix.txt", mat_hopfield, fmt="%d")

    return mat_hopfield


img1_mat = image2Matrix(img1, 100)
img2_mat = image2Matrix(img2, 100)
img3_mat = image2Matrix(img3, 100)


if __name__ == "__main__":

    cmap_nb = mcolors.LinearSegmentedColormap.from_list("noir_blanc", ["black", "white"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    imgs_mat = [img1_mat, img2_mat, img3_mat]

    for i, img_mat in enumerate(imgs_mat):
        np.savetxt(f"src/hopfield/data/image{i+1}_100x100_matrix.txt", img_mat, fmt="%d")

        plt.imshow(img_mat, cmap=cmap_nb, norm=norm)
        plt.savefig(f"docs/images/im{i+1}.pdf")
        plt.show()