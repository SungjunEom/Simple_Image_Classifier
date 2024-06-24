from datasets.dataset import *
from torch.utils.data import DataLoader
from configs.train_config import TrainConfigs
from streaming import StreamingDataset
from typing import Any
from streaming.base.format.mds.encodings import Encoding, _encodings

configs = TrainConfigs().parse()
batch_size = configs.batch_size

def make_loader(name):
    if name == 'Cifar-10':
        train_data_path = [
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_1',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_2',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_3',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_4',
        './datasets/data/cifar-10/cifar-10-batches-py/data_batch_5'
        ]
        test_data_path = './datasets/data/cifar-10/cifar-10-batches-py/test_batch'
  
        train_dataset = CifarDataset(data_path=train_data_path[0])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataset = CifarDataset(data_path=test_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

        return train_dataloader, test_dataloader
    
    elif name=='ImageNet.int8': # working on
        class uint8(Encoding):
            def encode(self, obj: Any) -> bytes:
                return obj.tobytes()

            def decode(self, data: bytes) -> Any:
                x=  np.frombuffer(data, np.uint8).astype(np.float32)
                return (x / 255.0 - 0.5) * 24.0

        _encodings["uint8"] = uint8

        remote_train_dir = "/workspace/Datasets/Images/vae_mds" # this is the path you installed this dataset.
        local_train_dir = "./local_train_dir"

        train_dataset = StreamingDataset(
            local=local_train_dir,
            remote=remote_train_dir,
            split=None,
            shuffle=True,
            shuffle_algo="naive",
            num_canonical_nodes=1,
            batch_size = 32
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            num_workers=2,
        )

        return train_dataloader
    
    elif name=='synthetic-sdxl':
        train_data_path = './datasets/data/train'
        test_data_path = './datasets/data/test'
        
        train_dataset = SyntheticDataset(train_data_path)
        test_dataset = SyntheticDataset(test_data_path)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

        return train_dataloader, test_dataloader