import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from MachineLearning import get_features, create_heatmap, get_images
from CustomDataset import CustomDataset
import torch

# model definition
# https://duchesnay.github.io/pystatsml/deep_learning/dl_mlp_mnist_pytorch.html
class TwoLayerMLP(nn.Module):

    def __init__(self, d_in, d_hidden, d_out):
        super(TwoLayerMLP, self).__init__()
        self.d_in = d_in

        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)

    def forward(self, X):
        print(X.shape)
        X = X.view(-1, self.d_in)
        X = self.linear1(X)
        return F.log_softmax(self.linear2(X), dim=1)


labels = []
# This represents the data we want to train with for just clarke county
# ****
imgFolder = '/Users/will/StreetViewProject/streetview_images/3392401699999999_-83259033' # This will be the path to the folder with the images for Clarke County
imgs, showableImages = get_images(imgFolder)
feats, imgs = get_features(imgs)
newFeats = []
for feat in feats:
    # Append the average life expectancy for Clarke County
    labels.append(torch.tensor(78.5))
    # Removing dict from feats and just keeping the tensor
    newFeats.append(feat['layer4'])
# **** 
# Repeat this block for each county

# Dataset of features and labels
# Feature corresponds to the tensor of the features for each image
# Label corresponds to the average life expectancy for the feature
dataset = CustomDataset(newFeats, labels)
# print(newFeats)
print(dataset)
# print('\nFirst iteration of data set: ', next(iter(dataset)), '\n')

# Will need to find a way to automatically split the data into training and testing and validation datasets
training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# model = TwoLayerMLP(d_in=(2048 * 9 * 9), d_hidden=100, d_out=1)
model = nn.Linear(9, 1)

loss_function = nn.MSELoss()
trainer = torch.optim.SGD(model.parameters(), lr=0.03)

  # Run the training loop
for epoch in range(0, 5): # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(training_loader, 0):

        # Get inputs
        inputs = data['feature']
        targets = data['label']

        l = loss_function(model(inputs), targets)

        # Zero the gradients
        trainer.zero_grad()

        # Perform backward pass
        l.backward()

        # Perform optimization
        trainer.step()



        


# Process is complete.
print('Training process has finished.')
    


# train_data = DataLoader(dataset=feats)
