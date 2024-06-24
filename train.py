from models.CNN import *
from datasets.dataset import *
from utils import *
from configs.train_config import TrainConfigs
import torch
from tqdm import tqdm
import wandb


configs = TrainConfigs().parse()
isWandb = configs.wandb
device = configs.device
lr = configs.lr
loss_fn = configs.loss_fn.to(device)
epochs = configs.epochs
batch_size = configs.batch_size

if isWandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project=configs.project_name,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": configs.lr,
        "architecture": "Small",
        "dataset": configs.dataset_name,
        "epochs": configs.epochs,
        "batch_size": configs.batch_size,
        }
    )

def main():
    '''
    Train a new model
    '''
    assert torch.cuda.is_available(), 'No GPU detected'
    torch.manual_seed(configs.seed)

    # torch.cuda.set_device(configs.device)
  
    model = Medium(num_class=configs.class_num).to(device)
    train_dataloader, test_dataloader = make_loader(configs.dataset_name)
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

        loss = 0
        n = 0
        for X, y in test_dataloader:
            model.eval()
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y)
            n += 1
        loss = loss / n
        print('Loss: ',loss)

        if isWandb:
            wandb.log({"eval": loss})

    torch.save(model.state_dict(), './naive.pt')





if __name__=='__main__':
    main()