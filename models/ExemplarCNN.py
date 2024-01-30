import torch
import torch.nn as nn


class Small(nn.Module):
    '''
    Excerpt from the original paper:
    A “small” network was used to evaluate the influence
    of different components of the augmentation procedure on classification performance. It consists of
    two convolutional layers with 64 filters each followed by a fully connected layer with 128 neurons.
    This last layer is succeeded by a softmax layer, which serves as the network output. 
    '''
    def __init__(self, activation=nn.ReLU(), num_class=50):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.maxpool = nn.MaxPool2d(3,2)
        self.linear1 = nn.Linear(7*7*64,128)
        self.linear2 = nn.Linear(128,num_class)
        self.dropout = nn.Dropout(0.5)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x,1,-1)
        x = self.dropout(self.linear1(x))
        x = self.linear2(x)
        return x
