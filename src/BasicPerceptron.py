import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from MachineLearning import get_features, create_heatmap, get_images
from CustomDataset import CustomDataset
import torch

# model definition
# https://duchesnay.github.io/pystatsml/deep_learning/dl_mlp_mnist_pytorch.html
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):

    def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(inp, out) for inp, out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum((out for inp, out in embedding_dim))
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


def prepare_data(path, label):
    """_summary_

    Args:
        path (string): The path to the folder containing the images for a given county
        label (double): The life expectancy or other value to predict for the given image in county

    Returns:
        list, list: Two lists corresponding to the features and labels for the given county
    """

    labels = []
    # This represents the data we want to train with for just clarke county
    # ****
    imgFolder = path  # This will be the path to the folder with the images for Clarke County
    imgs, showableImages = get_images(imgFolder)
    feats, imgs = get_features(imgs)
    newFeats = []
    for feat in feats:
        # Append the average life expectancy for Clarke County
        labels.append(label)
        # Removing dict from feats and just keeping the tensor
        newFeats.append(feat['layer4'])
    # ****
    # Repeat this block for each county

    return newFeats, labels

# Dataset of features and labels
# Feature corresponds to the tensor of the features for each image
# Label corresponds to the average life expectancy for the feature
# newFeats, labels = prepare_data('/Users/will/StreetViewProject/streetview_images/3392401699999999_-83259033', 78.5)
# dataset = CustomDataset(newFeats, labels)
# print('\nFirst iteration of data set: ', next(iter(dataset)), '\n')

# Will need to find a way to automatically split the data into training and testing and validation datasets
# training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# model = nn.Linear(9, 1)


def train(dataset, model):
    loss_function = nn.MSELoss()
    trainer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    training_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True)

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(training_loader, 0):

            # Get inputs
            inputs = data['feature']
            targets = data['label']

            # Forward pass
            outputs = model(inputs)

            # Loss computation
            print("outputs: ", outputs)
            print("targets: ", targets)
            l = loss_function(outputs, targets)

            # Zero the gradients
            trainer.zero_grad()

            # Perform backward pass
            l.backward()

            # Perform optimization
            trainer.step()

    # Process is complete.
    print('Training process has finished.')

    torch.save(model, 'trained_model.pt')

# train()

# train_data = DataLoader(dataset=feats)
