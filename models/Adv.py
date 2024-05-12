import torch
import torch.nn as nn


class Filter(nn.Module):
    '''
    Perturbation
    '''
    def __init__(self, size):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Filter().to(device='cuda:0')
    summary(model,(3, 32, 32))