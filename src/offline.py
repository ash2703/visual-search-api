from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from tqdm import tqdm
from src.features import FeatureExtractor

IMG_PATH = Path("./static/imgs")
FEATURE_PATH = Path("./static/features")


if __name__ == "__main__":
    feats = FeatureExtractor()
    for img_path in tqdm(sorted(IMG_PATH.glob("*.*"))):
        feature_path = FEATURE_PATH / (img_path.stem + ".npy")
        if not feature_path.is_file():
            img_feats = feats.extract(img_path)
            np.save(feature_path, img_feats)

    