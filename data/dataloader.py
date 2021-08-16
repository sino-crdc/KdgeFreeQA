from tqdm import tqdm
import json
import config.latentqa_cail2021 as config
from data.dataset import bqaDataset
from torch.utils.data import DataLoader
from preprocess.text_utils import pad_token, unk_token, bos_token, eos_token, PAD, UNK, BOS, EOS
from preprocess.text_utils import word_seg, init_tokenizer
import sys
import torch
from preprocess.vector import load_vectors, convert_to_vector
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np


def get_train_loader(bqa_train_path, batch_size):
    """
    get a DataLoader
    :param bqa_train_path: path to background-question-answer-train dataset
    :param batch_size
    :return: DataLoader: which gives cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask
    """
    'process text from file'
    cases = []
    questions = []
    with open(bqa_train_path, 'r', encoding='utf-8') as f:
        print('reading data from ' + bqa_train_path)
        count = 0  # the following loop is too long, for quick debug, we only use some samples for trial.
        for line in tqdm(f.readlines()):
            count += 1
            if count == 100:  # todo : we load 100 cases for debugging.
                break
            line = line[1:-2]
            case = []
            question = []
            for x in line.split(', {'):
                try:
                    if x[0] != '{':
                        x = '{' + x
                    y = json.loads(x)
                except Exception as e:
                    print(line)
                    print(x)
                    sys.exit(0)
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

    'which vocabulary'
    print('registering vocabulary')
    if config.train_cfg['use_char'] is True:
        vocab = config.train_cfg['vocab_char_path']
    else:
        vocab = config.train_cfg['vocab_word_path']

    'tokenization'
    print('tokenizing')
    if config.train_cfg['is_train_tokenized']:
        # todo : load tokenized cases and questions from files.
        pass
    else:
        init_tokenizer(vocab)
        cases = list(map(word_seg, cases))  # then cases=[['第', '一句', '话'], ..., ['第', 'N句', '话']]
        questions = list(map(word_seg, questions))  # then cases = [['第', '一个', '问题'], ..., ['第', 'N个', '问题']]

    'convert text to tensors'
    def collate_fn(examples):
        """
        convert text to tensor : tokenized_text -> glove --> tensor
        :param examples: [{'case': xxx, 'question': xxx}, ...], and case/question : ['第', '一句', '话']
        :return: dict : {'case': tensor, 'question': tensor}
        """
        # todo : BOS and EOS

        # which vector
        'registering vectors'
        if config.train_cfg['use_char'] is True:
            vector = config.train_cfg['vector_char_path']
        else:
            vector = config.train_cfg['vector_word_path']

        # load_vectors
        vectors = load_vectors(vector)

        # convert to tensors
        print('converting data to tensors and padding')
        lengths_case = torch.tensor([len(ex['case']) for ex in examples])
        cases = [torch.tensor(convert_to_vector(vectors, ex['case'])) for ex in examples]
        lengths_question = torch.tensor([len(ex['question']) for ex in examples])
        questions = [torch.tensor(convert_to_vector(vectors, ex['question'])) for ex in examples]
        cases = pad_sequence(cases, batch_first=config.train_cfg['batch_first'], padding_value=PAD)
        questions = pad_sequence(questions, batch_first=config.train_cfg['batch_first'], padding_value=PAD)
        #  we return a mask (cases != PAD), which masks the pads(=False), for question, too
        return cases, questions, lengths_case, lengths_question, cases != PAD, questions != PAD, examples

    'define dataset and dataloader'
    print('registering dataset')
    dataset = bqaDataset(cases, questions)  # in string_tokenized
    print('registering dataloader')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader
