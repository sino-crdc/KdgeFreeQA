from collections import defaultdict
from preprocess.text_utils import PAD, UNK, BOS, EOS, pad_token, unk_token, bos_token, eos_token
from tqdm import tqdm
import sys


def convert_to_vector(vectors, text):
    """
    convert one text to vectors
    :param vectors : dict, text : text to convert, in tokenized form, ['第', '一句', '话']
    :return: in list
    """
    if vectors is None:
        raise Exception('Oops, no vectors yet. Please load it in advance : load_vectors（path_to_vector')
    else:
        return [vectors[word] for word in text]  # if word is not in the vectors, vectors[word] will be UNK (as a defaultdict)


# def convert_special_token(vectors, which):
#     """
#     convert special token to vector
#     :param vectors: dict
#     :param which: which token, in str
#     :return: a vector
#     """
#     if vectors is None:
#         raise Exception('Oops, no vectors yet. Please load it in advance : load_vectors（path_to_vector')
#     else:
#         return vectors[which]


def change_special_tokens(lines):
    global UNK, PAD, BOS, EOS
    print('changing vectors for special tokens')

    for line in tqdm(lines):
        items = line.split(' ')
        key = items[0]
        if key == unk_token:
            value = list(map(float, items[1:]))
            UNK = value
        elif key == pad_token :
            value = list(map(float, items[1:]))
            PAD = value
        elif key == bos_token:
            value = list(map(float, items[1:]))
            BOS = value
        elif key == eos_token:
            value = list(map(float, items[1:]))
            EOS = value

    print('special tokens changed.')




def load_vectors(path_to_vector, vocab_size):
    """
    load vectors.
    :param path_to_vector: where is the vector(GloVe/ELMo/Word2Vec/etc.)
    :return: a dict {str: list}
    """
    global UNK, PAD, BOS, EOS

    if path_to_vector is not None:
        # load vectors
        with open(path_to_vector, 'r', encoding='utf-8') as f:
            print('loading data from ' + path_to_vector, end='...')
            lines = f.readlines()
            print('loaded.')
            change_special_tokens(lines)
            vectors = defaultdict(lambda: UNK)
            print('parsing vectors from ' + path_to_vector)
            for i, line in enumerate(tqdm(lines)):
                if i == vocab_size:
                    break
                items = line.split(' ')
                key = items[0]
                value = list(map(float, items[1:]))
                vectors[key] = value
            print('parsed')

        # todo : attention, some or all of these special tokens are not in the vocab ('in file but not in idx2word/word2idx' or 'not in file and not in idx2word/word2idx')!
        vectors[pad_token] = PAD
        vectors[unk_token] = UNK
        vectors[bos_token] = BOS
        vectors[eos_token] = EOS

        return vectors
    else:
        raise Exception('Oops! No vector file ! Please check the path to the vectors, GloVe, ELMo, Word2Vec, etc.')