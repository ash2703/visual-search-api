import os
from pathlib import Path

cwd = Path(os.getcwd())

class DefaultConfig:
    def __init__(self):
        self.IMG_PATH = Path(f"{cwd}/fashion_data/images")
        self.FEATURE_PATH = Path(f"{cwd}/fashion_data/features")
        self.MODEL_PATH = Path(f"{cwd}/weights/last.ckpt")

        self.IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
        self.FEATURE_DIM = 2048

