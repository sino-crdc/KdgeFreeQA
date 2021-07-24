import spacy

pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

extra_tokens = [pad_token, unk_token, bos_token, eos_token]

PAD = extra_tokens.index(pad_token)
UNK = extra_tokens.index(unk_token)
BOS = extra_tokens.index(bos_token)
EOS = extra_tokens.index(eos_token)

'''
@author: Yakun
'''
def word_seg(text):

    pass # text w/ space or array