import argparse

class DefaultConfigs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Models are saved here')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
        self.parser.add_argument('--seed', type=int, default=0, help='Seed')
        self.parser.add_argument('--device', type=int, default=0, help='Specify GPU num')

    def parse(self):
        self.conf = self.parser.parse_args()

        return self.conf