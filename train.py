import json
from config import latentqa_cail2021 as config
import torch
from data.dataloader import get_train_loader


DEVICE = torch.device('cuda') if config.train_cfg['use_gpu'] is True and torch.cuda.is_available() else 'cpu'


def loss():
    pass


def train(dataloader):
    print('training : get data from dataloader')
    for cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples in dataloader:
        pass




def main(bqa_train_path, batch_size, *args, **kwargs):
    dataloader = get_train_loader(bqa_train_path, batch_size)
    train(dataloader)


if __name__ == '__main__':
    # todo : optimization : config file can only be assigned in this main function, the others get the config as a parameter
    # todo : like this : main(cfgs, bqa_train_path, batch_size), etc.
    # parse command config, c.f. [argparse](https://zhuanlan.zhihu.com/p/56922793)
    bqa_train_path = config.train_cfg['bqa_train_path']
    batch_size = config.train_cfg['batch_size']
    main(bqa_train_path,
         batch_size)

