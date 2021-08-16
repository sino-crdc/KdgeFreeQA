import torch.nn as nn
import transformers
from torchtext.legacy.data import Field


class GloveEmbedding(nn.Module):
    def __init__(self):
        super(GloveEmbedding, self).__init__()
        # we use existing GloVe, please see it in preprocess/vector.py
        # until now, we don't train it by ourselves
        # todo : but it may need to be trained by ourselves, because the original glove vectors may have too many UNK.(to be verified)


class ELMoEmbedding(nn.Module):
    pass


class BertEmbedding(nn.Module):
    pass
