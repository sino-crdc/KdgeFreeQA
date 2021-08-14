import torch.nn as nn
import transformers
from torchtext.legacy.data import Field

class GloveEmbedding(nn.Module):
    def __init__(self):
        super(GloveEmbedding, self).__init__()
        pass  # until now, we don't train it by ourselves

    @classmethod
    def embed(cls, vector_file, text):
        pass




class ELMoEmbedding(nn.Module):
    pass


class BertEmbedding(nn.Module):
    pass
