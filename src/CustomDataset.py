from torch.utils.data import Dataset, DataLoader
from MachineLearning import get_features, create_heatmap, get_images
# from skimage import io, transform


imgFolder = '/Users/will/StreetViewProject/streetview_images/3392401699999999_-83259033'
imgs, showableImages = get_images(imgFolder)
feats, imgs = get_features(imgs)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    # Maybe use labels instead of features
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]
        sample = {"feature": feature, "label": label}

        # if (self.transform):
        #     sample = self.transform(sample)

        return sample