from models.CNN import *
from datasets.dataset import *
from utils import *
from configs.train_config import TrainConfigs
import torch
from tqdm import tqdm
import wandb
from safetensors.torch import save_file

configs = TrainConfigs().parse()
isWandb = configs.wandb
device = configs.device
lr = configs.lr
loss_fn = configs.loss_fn.to(device)
epochs = configs.epochs
batch_size = configs.batch_size

if isWandb:
    wandb.init(
        project=configs.project_name,
        config={
            "learning_rate": lr,
            "architecture": "Medium",
            "dataset": configs.dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
        }
    )

def main():
    '''
    Train a new model
    '''
    assert torch.cuda.is_available(), 'No GPU detected'
    torch.manual_seed(configs.seed)
  
    model = Medium(num_class=configs.class_num).to(device)
    train_dataloader, test_dataloader = make_loader(configs.dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                wandb.log({"loss": loss.item()})

        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        n = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                eval_loss += loss_fn(pred, y).item()
                n += 1
        avg_loss = eval_loss / n if n > 0 else 0
        print('Loss: ', avg_loss)

        if isWandb:
            wandb.log({"eval": avg_loss})

    save_file(model.state_dict(), './naive.safetensors')


if __name__=='__main__':
    main()
