import config.latentqa_cail2021 as config

def word2idx(path_to_vocab):
    """
    :param path_to_vocab:
    :return: a dict
    """
    words = []
    word2idx = {}
    with open(path_to_vocab, 'r', encoding='utf-8') as f:
        words = f.readlines()
    for i, word in enumerate(words):
        word2idx[word.strip()] = i
    return word2idx


def idx2word(path_to_vocab):
    """
    :param path_to_vocab:
    :return: a list
    """
    with open(path_to_vocab, 'r', encoding='utf-8') as f:
        words = f.readlines()
    for i, word in enumerate(words):
        words[i] = word.strip()
    return words


def text_to_idx(example, word2idx):
    return [word2idx[word] for word in example]


def idx_to_text(example, idx2word):
    return [idx2word[idx] for idx in example]


def idx_to_onehot(example, dim):
    return [[0 if i != idx else 1 for i in range(dim)] for idx in example]