from models.CNN import *
from datasets.dataset import *
from configs.train_config import TrainConfigs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

configs = TrainConfigs().parse()

if __name__ == '__main__':
    device = configs.device

    model = Medium(num_class=configs.class_num).to(device)
    state_dict = load_file('./naive.safetensors')
    model.load_state_dict(state_dict)

    print('Size of linear1: ',model.fc1.weight.shape)
    print('Rank of linear1:', torch.linalg.matrix_rank(model.fc1.weight))

    print('Size of linear2: ',model.fc2.weight.shape)
    print('Rank of linear2:', torch.linalg.matrix_rank(model.fc2.weight))