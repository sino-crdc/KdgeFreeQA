import tqdm
import json
import config.latentqa_cail2021 as config
from data.dataset import bqaDataset
from torch.utils.data import DataLoader
from preprocess.text_utils import pad_token, unk_token, bos_token, eos_token
from preprocess.text_utils import word_seg


def get_train_loader(bqa_train_path, batch_size):

    'process text from file'
    cases = []
    questions = []
    with open(bqa_train_path, 'r', encoding='utf-8') as f:
        print('reading from ' + bqa_train_path)
        for line in tqdm(f.readlines()):
            line = line[1:-2]
            case = []
            question = []
            for x in line.split(', '):
                y = json.loads(x)
                if 'question' in list(y.keys())[0]:
                    question.append(list(y.values())[0])
                content = list(y.values())[0]
                case.append(content)
            # move background to the first
            background = case.pop().strip()
            case.insert(0, background)
            case = ''.join(case)
            # one-case n-question = n samples with the same case
            for i in range(len(question)):
                cases.append(case)
            questions.extend(question)

    'load vocabulary'
    if config.train_cfg['use_char'] is True:
        vocab = config.train_cfg['vocab_char_path']
    else:
        vocab = config.train_cfg['vocab_word_path']

    'tokenization'
    # tokenizer = word_seg(vocab, text)


    'padding'


    'convert text to tensors'
    def collate_fn(examples):
        '''
        convert text to tensor : tokenized_text -> glove --> tensor
        :param examples: [‘第一句话’, '第二句话', '第三句话', ..., '第 bsz 句话']
        :return: dict : {'case': tensor, 'question': tensor}
        '''
        # attention for four special tokens
        # [bsz, lengths, dim]
        pass

    'define dataset and dataloader'
    dataset = bqaDataset(cases, questions)  # in string_tokenized
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
