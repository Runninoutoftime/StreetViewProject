import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # Linear layer
        self.fc1 = nn.Linear