import os
from pathlib import Path
from PIL import Image

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from tqdm import tqdm

from src.features import FeatureExtractor
from src.configs import config


if __name__ == "__main__":
    os.makedirs(config.FEATURE_PATH, exist_ok = True)
    feats = FeatureExtractor()
    for img_path in tqdm(sorted(config.IMG_PATH.glob("**/*"))):
        if img_path.suffix in config.IMAGE_EXTS:
            feature_path = config.FEATURE_PATH / (img_path.stem + ".npy")
            if not feature_path.is_file():
                img_feats = feats.extract(img_path)
                np.save(feature_path, img_feats)



    