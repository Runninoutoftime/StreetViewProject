from prepare_input import prepare
import torch
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
from torchvision import transforms
from imutils import paths
from matplotlib import pyplot as plt
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
import seaborn as sns
from render import show_images
from image_utils import center_crop, resize_shortest_edge

# Load the pretrained resnet model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)


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
    """Creates 4 heatmaps of the last layer for each image 

    Args:
        outs (_type_): _description_
        imgs (_type_): _description_
        show_heatmap (_type_): _description_
    """

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
    Processes the images in the given folder and returns a list of processed images
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


# # Example use
# imgFolder = '/Users/will/StreetViewProject/streetview_images/3392401699999999_-83259033'
# imgs, showableImages = get_images(imgFolder)
# feats, imgs = get_features(imgs)
# create_heatmap(feats, showableImages, True)
# print(feats[0]['layer4'])
# # Each feature is a (1x2048x9x9) tensor