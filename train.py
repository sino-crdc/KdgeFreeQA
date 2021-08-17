from config import latentqa_cail2021 as config
import torch
from data.dataloader import get_train_loader
import preprocess.vocab as vocab
from model.inference import LatentQA
import tensorboardX


DEVICE = torch.device('cuda') if config.train_cfg['use_gpu'] is True and torch.cuda.is_available() else 'cpu'


def loss():
    pass


def train(model_cfg, dataloader, idx2word, word2idx, batch_first=True):
    print('training...')
    model = LatentQA(model_cfg, idx2word, word2idx, batch_first)
    print('cuda is available ? ' + str(torch.cuda.is_available()))
    model.to(DEVICE)
    for num_step, (cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples) in enumerate(dataloader):
        probs = model(cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples, num_step)
        print(probs.shape)


def main(train_cfg, model_cfg, *args, **kwargs):
    if train_cfg['use_char'] is False:
        idx2word = vocab.idx2word(train_cfg['vocab_word_path'], train_cfg['vocab_size'])
        word2idx = vocab.word2idx(train_cfg['vocab_word_path'], train_cfg['vocab_size'])
    else:
        idx2word = vocab.idx2word(train_cfg['vocab_char_path'], train_cfg['vocab_size'])
        word2idx = vocab.word2idx(train_cfg['vocab_char_path'], train_cfg['vocab_size'])

    dataloader = get_train_loader(train_cfg)
    train(model_cfg, dataloader, idx2word, word2idx, train_cfg['batch_first'])


if __name__ == '__main__':
    # parse command config, c.f. [argparse](https://zhuanlan.zhihu.com/p/56922793)
    main(config.train_cfg, config.model_cfg)

