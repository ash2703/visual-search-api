from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np


model = VGG16(weights = "imagenet")

print(model.summary())

# dog1_feat = feats.extract(IMG_PATH / "dog-1.jpeg")
# dog2_feat = feats.extract(IMG_PATH / "dog-3.jpeg")
# cos_sim=cosine_similarity(dog1_feat.reshape(1,-1),dog2_feat.reshape(1,-1))
# print(f"Cosine Similarity between A and B:{cos_sim}")
# print(f"Cosine Distance between A and B:{1-cos_sim}")'

# from PIL import Image
# img = Image.open("./static/imgs/car-1.jpg")

# model = models.vgg16(pretrained=True)
# print(model)
# # remove last fully-connected layer
# new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# model.classifier = new_classifier

# transform = transforms.Compose([  
#     transforms.Resize(256),       
#     transforms.CenterCrop(224),   
#     transforms.ToTensor(),        
#     transforms.Normalize(         
#     mean=[0.485, 0.456, 0.406],   
#     std=[0.229, 0.224, 0.225]     
#     )])
# img_t = transform(img)


# model.eval()
# out = model(batch_t)
# print(out.shape)

# with open('./static/labels/imagenet_classes.txt') as f:
#   classes = [line.strip() for line in f.readlines()]

# _, index = torch.max(out, 1)

# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# print(classes[index[0]], percentage[index[0]].item())


# _, indices = torch.sort(out, descending=True)
# res = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
# print(res)