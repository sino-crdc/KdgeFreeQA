#coding:utf-8
from ltp import LTP
from config import latentqa_cail2021 as config
from torch.nn.utils.rnn import pad_sequence
import jieba
import time

# special tokens in str form.
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

# todo : how to delete the config, we want to import config only in train/etc. files.

dim = config.train_cfg['glove_dim']

tokenizer = None
if config.train_cfg['tokenizer'] == 'ltp':
    tokenizer = LTP(config.train_cfg['LTP_config'])
else:  # config.train_cfg['tokenizer'] == 'jieba'
    tokenizer = jieba


def init_tokenizer(vocab):
    global tokenizer
    if vocab is not None:
        print('initialize tokenizer')
        if config.train_cfg['tokenizer'] == 'ltp':
            tokenizer.init_dict(vocab, max_window=4)
        else: # config.train_cfg['tokenizer'] == 'jieba'
            tokenizer.load_userdict(vocab)


def word_seg(text):
    global tokenizer
    if config.train_cfg['tokenizer'] == 'ltp':
        if text is list:
            segment = tokenizer.seg(text)[0][0]
        else:
            segment = tokenizer.seg([text])[0][0]
    else: # config.train_cfg['tokenizer'] == 'jieba'
        if text is list:
            segment = list(tokenizer.cut(text[0]))
        else:
            segment = list(tokenizer.cut(text))
    return segment


def convert_to_one_hot(num, dim):
    return [0 if i != num else 1 for i in range(dim)]


# special tokens in 100d form.
# PAD = [0 for i in range(dim)]
UNK = [1 for i in range(dim)]
BOS = convert_to_one_hot(0, dim)
EOS = convert_to_one_hot(dim-1, dim)

# special tokens in int form.
special_token_list = [pad_token, unk_token, bos_token, eos_token]
PAD = special_token_list.index(pad_token)  # pad token must be a number, not list <-- torch.nn.utils.rnn.pad_sequence, but need to be masked later !!!
# UNK = special_token_list.index(unk_token)
# BOS = special_token_list.index(bos_token)
# EOS = special_token_list.index(eos_token)