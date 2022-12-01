import torch
from BasicPerceptron import FeedForwardNN, prepare_data, train
from CustomDataset import CustomDataset
from MachineLearning import get_images, get_features, create_heatmap

# model = TwoLayerMLP(d_in=(2048 * 9 * 9), d_hidden=100, d_out=1)

# model.load_state_dict(torch.load('trained_model.pt'))

## Train the model

train_model = True

if (train_model == True):
    # Dataset of features and labels
    # Feature corresponds to the tensor of the features for each image
    # Label corresponds to the average life expectancy for the feature
    newFeats, labels = prepare_data('/Users/will/StreetViewProject/streetview_images/', torch.tensor([78.5]))
    print("LEN:", len(newFeats))
    dataset = CustomDataset(newFeats, labels)
    print('\nFirst iteration of data set: ', next(iter(dataset)), '\n')


    model = FeedForwardNN(embedding_dim, len(newFeats), 1, [100, 50], 0.4)

    model.train()
    train(dataset, model)

    # Will need to find a way to automatically split the data into training and testing and validation datasets
    # training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


## Use the model

model = torch.load('trained_model.pt')

## Set the model to evaluation mode
# model.eval()

# img_feats = 

imgs_to_test = '/Users/will/StreetViewProject/streetview_images/33960272_-83296329'

imgs, showableImgs = get_images(imgs_to_test)
img_feats, imgs = get_features(imgs)

# newFeats, labels  prepare_data(imgs_to_test, 78.5)

# print(img_feats[0]['layer4'])
output = model(img_feats[0]['layer4'])
prediction = torch.argmax(output)

print('Prediction: ', prediction)
print('*****')
print('Output: ', output)
print('*****')
print('Actual: ', 78.5)