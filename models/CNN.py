import torch
import torch.nn as nn
import torch.nn.functional as F


class Small(nn.Module):
    '''
    For Cifar-10.
    '''
    def __init__(self, activation=nn.ReLU(), num_class=10):
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
    
class Medium(nn.Module):
    '''
    For the sdxl synthetic dataset.
    '''
    def __init__(self, activation=nn.ReLU(), num_class=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjusted input size
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x32x32
        
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer with raw logits
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Medium().to(device='cuda:0')
    summary(model,(3, 256, 256))