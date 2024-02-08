from models.CNN import *
from datasets.dataset import *
from configs.train_config import TrainConfigs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


configs = TrainConfigs().parse()
isWandb = configs.wandb


if isWandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="First run",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": configs.lr,
        "architecture": "Small",
        "dataset": "Cifar10",
        "epochs": configs.epochs,
        }
    )


def main():
    '''
    Train a new model
    '''
    assert torch.cuda.is_available(), 'No GPU detected'
    torch.manual_seed(configs.seed)


    # torch.cuda.set_device(configs.device)
    train_data_path = [
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_1',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_2',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_3',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_4',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_5'
    ]

    test_data_path = './datasets/data/cifar-10/cifar-10-batches-py/test_batch'

    device = configs.device
    lr = configs.lr
    loss_fn = configs.loss_fn.to(device)
    epochs = configs.epochs
    batch_size = configs.batch_size
    

    model = Small(num_class=configs.class_num).to(device)
    model.load_state_dict(torch.load('./naive.pt'))


    test_dataset = CifarDataset(data_path=test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(epochs):
        pass





if __name__=='__main__':
    main()