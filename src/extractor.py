import torch
from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        self.model = models.vgg16(pretrained = True)
        new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.classifier = new_classifier
        self.transform = transforms.Compose([  
                            transforms.Resize(256),       
                            transforms.CenterCrop(224),   
                            transforms.ToTensor(),        
                            transforms.Normalize(         
                            mean=[0.485, 0.456, 0.406],   
                            std=[0.229, 0.224, 0.225]     
                            )])
        self.model.eval()

    def extract(self, img_path: Path):
        assert isinstance(img_path, Path)
        img = Image.open(img_path)
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            feature = self.model(batch_t)[0]
        return feature / np.linalg.norm(feature)

if __name__ == "__main__":
    fe = FeatureExtractor()
    print("new model:\n", fe.model)