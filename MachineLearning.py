import random
import torch
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
from torchvision import transforms
from imutils import paths
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import numpy as np
from matplotlib import pyplot as plt
import cv2
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import seaborn as sns
from render import show_images
from image_utils import center_crop, resize_shortest_edge

# Load the pretrained resnet model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)



# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_img = preprocess(Image.open('/Users/will/StreetViewProject/panorama_images/33960272_-83296329.jpg')).unsqueeze(0)

train_nodes, eval_nodes = get_graph_node_names(model)
assert([t == e for t, e in zip(train_nodes, eval_nodes)])
# print(train_nodes)

return_nodes = ['layer1', 'layer2', 'layer3', 'layer4']

feature_extractor = create_feature_extractor(model, return_nodes)

with torch.no_grad():
    out = feature_extractor(preprocess(Image.open('/Users/will/StreetViewProject/panorama_images/33960272_-83296329.jpg')).unsqueeze(0))

imagePaths = sorted(list(paths.list_images('/Users/will/StreetViewProject/panorama_images')))

imgs = []
for imagePath in imagePaths:
    img = cv2.imread(imagePath)[..., ::-1]
    img = resize_shortest_edge(img, 384, interpolation='auto')
    img = center_crop(img, (384, 384))
    imgs.append(img)

# show_images(imgs, per_row=5, imsize=(5, 5))

from prepare_input import prepare

inps = [prepare(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for img in imgs]

with torch.no_grad():
    out = torch.softmax(model(inps[0]), -1)[0].numpy()


return_nodes = ['layer1', 'layer2', 'layer3', 'layer4']

feat_ext = create_feature_extractor(model, return_nodes=return_nodes)

with torch.no_grad():
    out = feat_ext(inps[0])

fig, ax = plt.subplots(4, 5, figsize=(25, 20))

# Pick 4 random feature maps from each layer
for i, layer in enumerate(return_nodes):
    feat_maps = out[layer].numpy().squeeze(0)
    # print(list(feat_maps)[0])
    feat_maps = random.sample(list(feat_maps), 4)
    ax[i][0].imshow(imgs[0])
    ax[i][0].set_xticks([])
    ax[i][0].set_yticks([])
    for j, feat_map in enumerate(feat_maps):
        sns.heatmap(feat_map, ax=ax[i][j+1], cbar=False)
        ax[i][j+1].set_xticks([])
        ax[i][j+1].set_yticks([])
        ax[i][j+1].set_title(f'{layer}: ({feat_map.shape[0]} x {feat_map.shape[1]})')
plt.show()