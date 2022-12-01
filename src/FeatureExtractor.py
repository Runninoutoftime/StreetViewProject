import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck
from PIL import Image
from torchvision import transforms
from imutils import paths
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
import numpy as np
import requests
import cv2

# Load the pretrained resnet model
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.fc = torch.nn.Identity()
# 

# Grad cam setup
# target_layers = [model.layer4[-1]]
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False) # Change if on GPU
# targets = [ClassifierOutputTarget(281)] # What does this 281 do lol?


# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Based on:
# https://stackoverflow.com/a/62118437
def get_final_features(input_image):
    """Get the features of an image

    Args:
        input_image (string): The path to the image to get features from

    Returns:
        tensor(float[1, 2048]): A tensor of size 1x2048 containing the features of the image
    """

    # Sets last layer to identity
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()

    input_image = Image.open(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    features = model(input_batch)
    print(features[0])#[1..]) # Features is a 1x2048 tensor
    print(features[0].shape)
    return features

def get_layer_3_features(input_image):
    """Get the features of an image from the last function in the third layer of ResNet50

    Args:
        input_image (string): The path to the image to get features from

    Returns:
        tensor(float[1, 2048]): A tensor of size 1x2048 containing the features of the image
    """

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.layer4 = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()

    input_image = Image.open(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    features = model(input_batch)
    print(features[0])#[1..]) # Features is a 1x2048 tensor
    print(features[0].shape)
    return features

def get_layer_2_features(input_image):
    """Get the features of an image from the last function in the second layer of ResNet50

    Args:
        input_image (string): The path to the image to get features from

    Returns:
        tensor(float[1, 2048]): A tensor of size 1x2048 containing the features of the image
    """

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.layer3 = torch.nn.Identity()
    model.layer4 = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()

    input_image = Image.open(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    features = model(input_batch)
    print(features)
    print(features[0])#[1..]) # Features is a 1x2048 tensor
    print(features[0].shape)
    return features


def get_layer_1_features(input_image):
    """Get the features of an image from the last function in the first layer of ResNet50

    Args:
        input_image (string): The path to the image to get features from

    Returns:
        tensor(float[1, 2048]): A tensor of size 1x2048 containing the features of the image
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.layer2 = torch.nn.Identity()
    model.layer3 = torch.nn.Identity()
    model.layer4 = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()

    input_image = Image.open(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    features = model(input_batch)
    print(features[0])#[1..]) # Features is a 1x2048 tensor
    print(features[0].shape)
    return features

def eval_images(image_folder, get_features):
    """Evaluate a folder of images and return a list of features

    Args:
        image_folder (string): The folder containing images to evaluate
        get_features (string): The layer to get features from. Can be either layer_1, layer_2, layer_3, or final
    """
    imagePaths = sorted(list(paths.list_images(image_folder)))
    images = []

    if (get_features == 'layer_1'):
        feature = get_layer_1_features
    elif (get_features == 'layer_2'):
        feature = get_layer_2_features
    elif (get_features == 'layer_3'):
        feature = get_layer_3_features
    elif (get_features == 'final'):
        feature = get_final_features


    feats = []
    for imagePath in imagePaths:
        feats.append(feature(imagePath))
    
    return feats


# eval_images('/Users/will/StreetViewProject/panorama_images')

