import pickle
import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
# from streaming.base.format.mds.encodings import Encoding, _encodings
# from streaming import StreamingDataset


class CifarDataset(Dataset):
    def __init__(self, data_path='./data/cifar-10/cifar-10-batches-py/data_batch_1'):
        self.data_path = data_path
        with open(self.data_path, 'rb') as fo:
            self.file = pickle.load(fo, encoding='bytes')

        self.images = self.file[b'data']
        self.images = self.images.reshape(-1,3,32,32)
        self.images = torch.tensor(self.images/255, dtype=torch.float32)
        self.labels = self.file[b'labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
        

class SyntheticDataset(Dataset):
    def __init__(self, data_path='./data/train'):
        self.data_path = data_path

        self.male = os.listdir(self.data_path+'/male')
        self.female = os.listdir(self.data_path+'/female')

        self.male = [self.data_path+'/male/'+s for s in self.male]
        self.female = [self.data_path+'/female/'+s for s in self.female]

        self.labels = [0] * len(self.male) + [1] * len(self.female)
        self.images = self.male + self.female

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])   
        transform = transforms.ToTensor()
        image = transform(image) # (3, 1024, 1024)
        # image = image.permute(1, 2, 0) # if this is uncommented, the tensor shape will be (3, 1024, 1024).
        
        # Shrink the image into (3, 256, 256) to avoid OOM
        resize_transform = transforms.Resize((256, 256))

        # Apply the transformation
        # Add batch dimension before transform and remove it after
        image = resize_transform(image.unsqueeze(0)).squeeze(0)
        
        return image, self.labels[idx]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wow = SyntheticDataset()

    image, label = wow.__getitem__(2)
    
    print(image.shape)
