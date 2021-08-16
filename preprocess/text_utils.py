from ltp import LTP
from config import latentqa_cail2021 as config
from torch.nn.utils.rnn import pad_sequence

# special tokens in str form.
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

dim = config.train_cfg['glove_dim']

ltp = LTP(config.train_cfg['LTP_config'])


def init_tokenizer(vocab):
    if vocab is not None:
        ltp.init_dict(vocab, max_window=4)


def word_seg(text):
    if text is list:
        segment = ltp.seg(text)[0]
    else:
        segment = ltp.seg([text])[0]
    return segment[0]


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