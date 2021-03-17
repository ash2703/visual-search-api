import os
from pathlib import Path

cwd = Path(os.getcwd())

class DefaultConfig:
    def __init__(self):
        self.IMG_PATH = Path(f"{cwd}/fashion_data/images")
        self.FEATURE_PATH = Path(f"{cwd}/fashion_data/features")
        self.IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

