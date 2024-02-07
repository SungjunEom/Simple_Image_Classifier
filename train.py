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
        "dataset": "STL-10",
        "epochs": configs.epochs,
        }
    )

def main():
    '''
    Trains a new model
    '''
    assert torch.cuda.is_available(), 'No GPU detected'
    torch.manual_seed(configs.seed)
    # torch.cuda.set_device(configs.device)
    data_path = configs.dataset_path
    device = configs.device
    lr = configs.lr
    loss_fn = configs.loss_fn.to(device)
    epochs = configs.epochs
    batch_size = configs.batch_size
    
    model = Small(num_class=configs.class_num).to(device)
    train_dataset = TrainDataset(data_path=data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(epochs):
        print('Epoch:', epoch)
        model.train()
        for X, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if isWandb:
                wandb.log({"loss": loss})




if __name__=='__main__':
    main()