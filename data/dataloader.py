from tqdm import tqdm
import json
from data.dataset import cqaDataset
from torch.utils.data import DataLoader
from preprocess.text_utils import pad_token, unk_token, bos_token, eos_token, PAD, UNK, BOS, EOS
from preprocess.text_utils import word_seg, init_tokenizer
import sys
import torch
from preprocess.vector import load_vectors, convert_to_vector
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

def get_train_loader(train_cfg):
    """
    get a DataLoader

    :return: DataLoader: which gives cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask
    """

    # which vector
    'registering vectors'
    if train_cfg['use_char'] is True:
        vector = train_cfg['vector_char_path']
    else:
        vector = train_cfg['vector_word_path']

    # load_vectors
    vectors = load_vectors(vector, train_cfg['vocab_size'])

    'which vocabulary'
    if train_cfg['use_char'] is True:
        vocab = train_cfg['vocab_char_path']
    else:
        vocab = train_cfg['vocab_word_path']

    'initialize tokenizer'
    if train_cfg['init_tokenizer_by_own_vocab']:
        init_tokenizer(vocab)

    'convert text to tensors'
    def collate_fn(examples):
        """
        convert text to tensor : tokenized_text -> glove --> tensor
        :param examples: [{'case': xxx, 'question': xxx}, ...], and case/question : ['第', '一句', '话']
        :return: dict : {'case': tensor, 'question': tensor}
        """
        # todo : BOS and EOS
        # convert to tensors
        lengths_case = torch.tensor([len(ex['case']) for ex in examples])
        cases = [torch.tensor(convert_to_vector(vectors, ex['case'])) for ex in examples]
        lengths_question = torch.tensor([len(ex['question']) for ex in examples])
        questions = [torch.tensor(convert_to_vector(vectors, ex['question'])) for ex in examples]
        cases = pad_sequence(cases, batch_first=train_cfg['batch_first'], padding_value=PAD)
        questions = pad_sequence(questions, batch_first=train_cfg['batch_first'], padding_value=PAD)
        #  we return a mask (cases != PAD), which masks the pads(=False), for question, too
        return cases, questions, lengths_case, lengths_question, cases != PAD, questions != PAD, examples

    'define dataset and dataloader'
    print('register dataset')
    dataset = cqaDataset(train_cfg)  # in string_tokenized
    print('register dataloader')
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)

    return dataloader
