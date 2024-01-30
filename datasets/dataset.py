from .stl_preprocessing import *
from configs.train_config import *
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import numpy as np


config = TrainConfigs().parse()
device = config.device

class TrainDataset(Dataset):
    def __init__(self, k=150, n=8000, data_path='./data/stl10_binary/unlabeled_X.bin'):
        self.config = TrainConfigs().parse()
        self.data_path = data_path
        self.all_images = read_all_images(self.data_path)
        self.img_w = self.all_images.shape[-1]
        self.img_h = self.all_images.shape[-2]
        self.k = k
        self.n = n

        self.labels = [i for i in range(0, n)]
        np.random.shuffle(self.all_images)
        self.all_images = self.all_images[:n]
        self.all_images = torch.Tensor(self.all_images).to(device)

        #CPU
        # self.transform = transforms.Compose([
        #     v2.ColorJitter(),
        #     v2.RandomRotation(20),
        #     v2.RandomResize(int(self.img_w * 0.7), int(self.img_w * 1.4)),
        #     v2.RandomCrop((32,32))
        # ])

        #GPU
        self.transform = torch.nn.Sequential(
            v2.ColorJitter(),
            v2.RandomRotation(20),
            v2.RandomResize(int(self.img_w * 0.7), int(self.img_w * 1.4)),
            v2.RandomCrop((32,32))
        ).to(device)


    def __len__(self):
        return self.n * self.k

    def __getitem__(self, idx):
        instance = idx // self.k
        image = self.transform(self.all_images[instance])
        return image, self.labels[instance]
        
    
class TestDataset(Dataset):
    def __init__(
            self, 
            data_path='./data/stl10_binary/test_X.bin',
            label_path='./data/stl10_binary/test_y.bin'
            ):
        self.data_path = data_path
        self.label_path = label_path
        self.all_images = read_all_images(self.data_path)

    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, idx):
        return self.all_images[idx]


if __name__ == '__main__':
    print('train:',read_all_images('./data/stl10_binary/train_X.bin').shape)
    print('test:',read_all_images('./data/stl10_binary/test_X.bin').shape)
    print('unlabeled:',read_all_images('./data/stl10_binary/unlabeled_X.bin').shape)