from ltp import LTP
from config import latentqa_cail2021 as config

pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

extra_tokens = [pad_token, unk_token, bos_token, eos_token]

PAD = extra_tokens.index(pad_token)
UNK = extra_tokens.index(unk_token)
BOS = extra_tokens.index(bos_token)
EOS = extra_tokens.index(eos_token)


def word_seg(vocab, text):
    ltp = LTP(config.train_cfg['LTP_config'])
    ltp.init_dict(vocab, max_window=4)
    if text is list:  # ["他叫汤姆去拿外衣。"]
        return ltp.seg(text)[0]  # ['他', '叫', '汤姆', '去', '拿', '外衣', '。']
    else:  # "他叫汤姆去拿外衣。"
        return ltp.seg([text])[0]  # ['他', '叫', '汤姆', '去', '拿', '外衣', '。']
