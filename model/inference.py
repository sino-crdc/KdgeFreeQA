import torch
import torch.nn as nn
from model.layers import Encoder, Decoder, context_vector, AttentionLayer
from model.selector import Selector
from preprocess.vocab import idx2word, word2idx


class LatentQA(nn.Module):
    def __init__(self, config, idx2word, word2idx, batch_first=True):
        super(LatentQA, self).__init__()
        self.config = config
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.batch_first = batch_first
        self.encoder = Encoder(self.config['embedding_dim'], self.config['encoder_dim'], self.config['num_encoder_layers'], batch_first=batch_first)
        self.decoder = Decoder(self.config['embedding_dim'], self.config['decoder_dim'], self.config['num_decoder_layers'], batch_first=batch_first)
        self.attention_layer = AttentionLayer(2 * self.config['encoder_dim'], 2 * self.config['encoder_dim'], self.config['decoder_dim'], batch_first)
        self.selector = Selector(idx2word, word2idx, self.config['latent_dim'], self.config['num_dist_word_selected'], self.config['temperature_initial'], self.config['mode'], batch_first)


    # cases and questions in vector form.
    def forward(self, cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples, num_step):
        print('encode')
        encoded_cases, encoded_questions = self.encoder(cases, questions)
        print('decode')
        decoded = self.decoder(cases, questions)
        print('attention map')
        att_c, att_q = self.attention_layer(encoded_cases, encoded_questions, decoded)
        print('context vector')
        context_c = context_vector(att_c, encoded_cases)
        context_q = context_vector(att_q, encoded_questions)
        print('stochastic selector network')
        probs = self.selector.forward(torch.cat([cases, questions], dim=1), att_c, att_q, context_c, context_q, decoded, examples, num_step)
        return probs


# """
# the dimensional analysis is very difficult, but this main function may help you.
# """
# # todo : Oops, the following example for dimensional analysis is NOT correct !
# # data construction
# array = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[11, 22, 33, 44], [55, 66, 77, 88]], [[1, 2, 3, 4], [5, 6, 7, 8]],
#          [[11, 22, 33, 44], [55, 66, 77, 88]]]
# examples = [{'case': ['第', '一句'], 'question': ['第', '一问']},
#     {'case': ['第', '二句'], 'question': ['第', '二问']},
#     {'case': ['第', '三句'], 'question': ['第', '三问']},
#     {'case': ['第', '四句'], 'question': ['第', '四问']}]
#
# data1 = torch.tensor(array, dtype=torch.float32)  # [4, 2, 4]
# data2 = torch.tensor(array, dtype=torch.float32)  # [4, 2, 4]
#
# # layers: from encoder to context vector.
# encoder = Encoder(4, 10, 1, True)
# cases, questions = encoder(data1, data2)  # cases = [4, 2, 20]
# decoder = Decoder(20, 10, 1, True)
# decoded = decoder(cases, questions)  # [4, 4, 10]
#
# # cases = nn.Linear(20, 10, bias=False)(cases)  # [4, 2, 10]
# # decoded = nn.Linear(10, 10, bias=True)(decoded)  # [4, 4, 10]
# # decodeds = torch.split(decoded, 1, dim=1)  # [4, 1, 10], [4, 1, 10], [4, 1, 10], [4, 1, 10]
# # print(decodeds[0].shape)
# # stack = []
# # for each in decodeds:
# #     dec, _ = torch.broadcast_tensors(each, cases)  # [4, 2, 10]
# #     print(decodeds[0].shape)
# #     print(dec.shape)
# #     activated = torch.tanh(dec + cases)
# #     logis = nn.Linear(10, 1, bias=False)(activated).squeeze()   # [4, 2]
# #     # print(decoded.shape[2])
# #     at = nn.Softmax(dim=1)(logis)
# #     stack.append(at)
# # print(decoded[0].shape)
# # att = torch.stack(stack, dim=2)
# # print(att.shape)  # [4, 2, 4]
#
# attention_layer = AttentionLayer(20, 20, 10)
# att_c, att_q = attention_layer(cases, questions, decoded)
# # print(att_q.shape)
# context_c = context_vector(att_c, cases)
# context_q = context_vector(att_q, questions)
#
# # snn : stochastic selection network
# idx2word = idx2word(config.train_cfg['vocab_word_path'])
# word2idx = word2idx(config.train_cfg['vocab_word_path'])
# num_step = 1  # the temperature is gradually annealed over the course of training.
# selector = Selector(idx2word, word2idx)
# # the following is selector.forward
# probs = selector.forward(torch.cat([cases, questions], dim=1), att_c, att_q, context_c, context_q, decoded, examples, num_step)
# print(probs.shape)






