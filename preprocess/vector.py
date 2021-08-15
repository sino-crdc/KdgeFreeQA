import config.latentqa_cail2021 as config
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


def load_vectors(path_to_vector):
    """
    load vectors.
    :param path_to_vector: where is the vector(GloVe/ELMo/Word2Vec/etc.)
    :return: a dict {str: list}
    """
    vectors = defaultdict(lambda: UNK)

    if path_to_vector is not None:
        # load vectors
        with open(path_to_vector, 'r', encoding='utf-8') as f:
            print('loading vectors from ' + path_to_vector)
            count = 0
            for line in tqdm(f.readlines()):
                count = count + 1
                if count == 10: # todo : we load 10 vectors for debugging.
                    break
                items = line.split(' ')
                key = items[0]
                value = list(map(float, items[1:]))
                vectors[key] = value
    else:
        raise Exception('Oops! No vector file ! Please check the path to the vectors, GloVe, ELMo, Word2Vec, etc.')

    vectors[pad_token] = PAD
    vectors[unk_token] = UNK
    vectors[bos_token] = BOS
    vectors[eos_token] = EOS

    return vectors