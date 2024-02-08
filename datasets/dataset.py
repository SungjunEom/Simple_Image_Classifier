import pickle
from torch.utils.data import Dataset
import torch
import numpy as np



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
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    wow = CifarDataset()

    print(wow.images.shape)
    print(len(wow.labels))

    data = wow.images[2]
    data = np.transpose(data, (1, 2, 0))
    im = Image.fromarray(data)
    im.save('wow.jpeg')