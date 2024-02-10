from models.CNN import *
from datasets.dataset import *
from configs.train_config import TrainConfigs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pyhessian import hessian


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
        project="Simple Image Classifier",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": configs.lr,
        "architecture": "Small",
        "dataset": "Cifar10",
        "epochs": configs.epochs,
        "batch_size": configs.batch_size,
        }
    )

def sharpness(model, X, y):

    hessian_comp = hessian(model, loss_fn, data=(X, y), cuda=True)
    top_eigenvalue, _ = hessian_comp.eigenvalues(top_n=1)

    return top_eigenvalue

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
  
    model = Small(num_class=configs.class_num).to(device)

    train_dataset = CifarDataset(data_path=train_data_path[0])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = CifarDataset(data_path=test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    


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

        sharp = sharpness(model, X, y)

        if isWandb:
            wandb.log({"eval": loss})
            wandb.log({"sharpness": sharp[0]})

    torch.save(model.state_dict(), './naive.pt')





if __name__=='__main__':
    main()