from models.CNN import *
from datasets.dataset import *
from configs.train_config import TrainConfigs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


configs = TrainConfigs().parse()

if __name__ == '__main__':
    device = configs.device

    model = Small(num_class=configs.class_num).to(device)
    model.load_state_dict(torch.load('./naive.pt'))

    print('Size of linear1: ',model.linear1.weight.shape)
    print('Rank of linear1:', torch.linalg.matrix_rank(model.linear1.weight))

    print('Size of linear2: ',model.linear2.weight.shape)
    print('Rank of linear2:', torch.linalg.matrix_rank(model.linear2.weight))