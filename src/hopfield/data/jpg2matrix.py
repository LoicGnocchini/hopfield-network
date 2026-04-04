from pathlib import Path
import numpy as np
from zipfile import ZipFile
from PIL import Image


def exctract_celebA():
    zip_path = Path("src/hopfield/data/img_align_celeba.zip")
    out_dir = Path("src/hopfield/data/celeba_subset")
    out_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zf:
        jpg_files = [name for name in zf.namelist() if name.endswith(".jpg")]
        
        for name in jpg_files[:800]:
            zf.extract(name, out_dir)



def jpg2matrix():
    # Dossier où sont les images CelebA
    input_dir = Path("src/hopfield/data/celeba_subset/img_align_celeba")

    # Nombre d'images voulues
    n_images = 800

    # Taille finale
    size = (100, 100)

    # Prend les 800 premières images jpg
    image_paths = sorted(input_dir.glob("*.jpg"))[:n_images]

    # Tableau final : (800, 100, 100)
    X = np.empty((len(image_paths), size[1], size[0]), dtype=np.int8)

    for k, path in enumerate(image_paths):
        img = Image.open(path)

        # 1. gris
        img = img.convert("L")

        # 2. resize 100x100
        img = img.resize(size)

        # 3. dithering noir/blanc
        img = img.convert("1")   # Floyd-Steinberg par défaut

        # 4. image -> numpy
        arr = np.array(img, dtype=np.uint8)

        # 5. convertir en -1 / +1
        # selon Pillow, en mode "1" on obtient du noir/blanc ; ici on mappe 0 -> -1, sinon +1
        arr = np.where(arr == 0, -1, 1).astype(np.int8)

        X[k] = arr
    return X


if __name__ == "__main__":

    exctract_celebA()
    X = jpg2matrix()
    P = X.reshape(X.shape[0], -1)
    np.save("src/hopfield/data/celeba_800_100x100_pm1_flat.npy", P)