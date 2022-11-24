from prepare_input import prepare
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


def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(284),
    transforms.CenterCrop(284),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# test_img = preprocess(Image.open(
#     '/Users/will/StreetViewProject/panorama_images/33960272_-83296329.jpg')).unsqueeze(0)

# # cv2.imshow("Title", torch.squeeze(test_img))

# train_nodes, eval_nodes = get_graph_node_names(model)
# assert ([t == e for t, e in zip(train_nodes, eval_nodes)])
# # print(train_nodes)

# return_nodes = ['layer1', 'layer2', 'layer3', 'layer4']

# feature_extractor = create_feature_extractor(model, return_nodes)

# with torch.no_grad():
#     out = feature_extractor(preprocess(Image.open(
#         '/Users/will/StreetViewProject/panorama_images/33960272_-83296329.jpg')).unsqueeze(0))

# imagePaths = sorted(list(paths.list_images(
#     '/Users/will/StreetViewProject/panorama_images')))

# imgs = []
# for imagePath in imagePaths:
#     img = cv2.imread(imagePath)[..., ::-1]
#     img = resize_shortest_edge(img, 284, interpolation='auto')
#     img = center_crop(img, (284, 284))
#     imgs.append(img)

# # show_images(imgs, per_row=5, imsize=(5, 5))


# inps = [prepare(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         for img in imgs]

# with torch.no_grad():
#     out = torch.softmax(model(inps[0]), -1)[0].numpy()


# return_nodes = ['layer1', 'layer2', 'layer3', 'layer4']

# feat_ext = create_feature_extractor(model, return_nodes=return_nodes)

# with torch.no_grad():
#     out = feat_ext(inps[0])

# fig, ax = plt.subplots(4, 5, figsize=(25, 20))

# # Pick 4 random feature maps from each layer
# feat_maps = out['layer4'].numpy().squeeze(0)
# # 2048x12x12
# feat_list = []
# for j in range(0, 4):
#     feat_list.append(list(feat_maps)[2044 + j])
# feat_maps = feat_list

# cv2.imshow("title", imgs[0])

# ax[0][0].imshow(imgs[0])
# ax[0][0].set_xticks([])
# ax[0][0].set_yticks([])
# for j, feat_map in enumerate(feat_maps):
#     sns.heatmap(feat_map, ax=ax[0][j+1], cbar=False)
#     ax[0][j+1].set_xticks([])
#     ax[0][j+1].set_yticks([])
#     ax[0][j+1].set_title(f'layer4: {j}')
# plt.show()


def get_features(imgList):
    """Takes an image and retunrs the features of the image

    Args:
        img (_type_): A path to an image

    Returns:
        _type_: Features of the image
    """
    feat_ext = create_feature_extractor(model, return_nodes=['layer4'])

    outs = []
    for input in imgList:
        with torch.no_grad():
            out = feat_ext(input)
            outs.append(out)

    return outs, imgList


def create_heatmap(outs, imgs, show_heatmap):

    fig, ax = plt.subplots(len(outs), 5, figsize=(25, 20))

    for i, img in enumerate(imgs):
        feat_maps = outs[i]['layer4'].numpy().squeeze(0)
        # 2048x12x12
        feat_list = []
        for j in range(0, 4):
            feat_list.append(list(feat_maps)[2044 + j])
        feat_maps = feat_list

        ax[i][0].imshow(imgs[i])
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        for j, feat_map in enumerate(feat_maps):
            sns.heatmap(feat_map, ax=ax[i][j+1], cbar=False)
            ax[i][j+1].set_xticks([])
            ax[i][j+1].set_yticks([])
            ax[i][j+1].set_title(f'layer4: {j}')

    if show_heatmap:
        plt.show()

def get_images(path):
    """
    Processes the images in the path and returns a list of processed images
    """
    imagePaths = sorted(list(paths.list_images(path)))
    
    images = []
    showableImgs = []
    for image in imagePaths:
        # images.append(cv2.imread(image)[..., ::-1])
        img = cv2.imread(image)[..., ::-1]
        img = resize_shortest_edge(img, 284, interpolation='auto')
        img = center_crop(img, (284, 284))
        input = prepare(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        images.append(input)
        showableImgs.append(img)

    return images, showableImgs

# img = Image.open(
#     '/Users/will/StreetViewProject/panorama_images/33960272_-83296329.jpg')
# feats = get_features(img)
# create_heatmap(img, feats, True)

imgFolder = '/Users/will/StreetViewProject/streetview_images/3392401699999999_-83259033'
imgs, showableImages = get_images(imgFolder)
feats, imgs = get_features(imgs)
create_heatmap(feats, showableImages, True)


# 1 Get all images from folder
# 2 Get features of each image
# 3 Create heatmap for each image