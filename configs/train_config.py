from .default import DefaultConfigs
import torch
import torch.nn as nn

class TrainConfigs(DefaultConfigs):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--n', type=int, default=1, help='The number of samples. N is in [50, 32000].')
        self.parser.add_argument('--k', type=int, default=1, help='The number of transformations. K is in [1, 300].')
        self.parser.add_argument('--epochs', type=int, default=50, help='epochs')
        self.parser.add_argument('--isTrain', type=bool, default=True, help='Is train?')
        self.parser.add_argument('--dataset_path', type=str, default='./datasets/data/', help='Path of target data')
        self.parser.add_argument('--dataset_name', type=str, default='synthetic-sdxl', help='Path of target data')
        self.parser.add_argument('--class_num', type=int, default=10, help='The number of surrogate classes')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        self.parser.add_argument('--wandb', type=bool, default=False, help='wandb.ai')
        self.parser.add_argument('--project_name', type=str, default='SimpleImageClassifier', help='Project name for wandb.ai')

    def parse(self):
        self.conf = self.parser.parse_args()
        self.conf.device = torch.device('cuda:0' if torch.cuda.is_available() else print('No GPU'))
        self.conf.loss_fn = nn.CrossEntropyLoss()

        return self.conf