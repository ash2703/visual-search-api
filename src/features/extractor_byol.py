from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

from ..models import BYOL, SelfSupervisedLearner


class FeatureExtractor:
    def __init__(self):
        IMAGE_SIZE = 256
        state_dict = torch.load("src/weights/epoch=14-step=135584.ckpt", 
                            map_location = torch.device('cpu')
                        )
        
        resnet = models.resnet50(pretrained=False)
        self.model = SelfSupervisedLearner(
                        resnet,
                        image_size = IMAGE_SIZE,
                        hidden_layer = 'avgpool',
                        projection_size = 256,
                        projection_hidden_size = 4096,
                        moving_average_decay = 0.99
                        )
        self.model.load_state_dict(state_dict["state_dict"])
        self.transform = transforms.Compose([  
                            transforms.Resize(IMAGE_SIZE),       
                            transforms.CenterCrop(IMAGE_SIZE),   
                            transforms.ToTensor(), 
                            transforms.Lambda(self.expand_greyscale)
                            ])
        self.model.eval()

    def expand_greyscale(self, t):
        return t.expand(3, -1, -1)

    def extract(self, img_path: Path):
        assert isinstance(img_path, Path)
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            _, embedding = self.model.learner(batch_t, return_embedding = True)  # outputs projections, embeddings
        return embedding / np.linalg.norm(embedding) 
    

if __name__ == "__main__":
    fe = FeatureExtractor()
    print("new model:\n", fe.model)