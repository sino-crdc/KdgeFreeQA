import json
from config import latentqa_cail2021 as config
from data.dataset import bqaDataset
from torch.utils.data import DataLoader

def loss():
    pass


def train():
    pass


def main():
    dataset = bqaDataset(config.train_cfg['bqa_train_path'])
    DataLoader()


if __name__ == '__main__':
    # parse command config, c.f. [argparse](https://zhuanlan.zhihu.com/p/56922793)
    main()

