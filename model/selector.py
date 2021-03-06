import torch.nn as nn
import torch
import sys
from preprocess.vocab import text_to_idx, idx_to_onehot
import numpy as np


class Selector(nn.Module):
    def __init__(self, idx2word, word2idx, latent_dim, num_dist_word_selected, temperature_initial, mode='gaussian', batch_first=True):
        """
        stochastic selector network
        :param idx2word: an array.
        :param word2idx: a dict.
        :param latent_dim: latent representation dimension.
        :param num_dist_word_selected: the number of distributions which a new word is selected from.
        :param temperature_initial: initial temperature for continuous softmax function (in place of argmax) in Gumbel-Max trick for inference of discrete latent variable.
        :param mode: parameterized distribution for latent representation.
        :param batch_first
        """
        super(Selector, self).__init__()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.mode = mode
        self.latent_dim = latent_dim
        self.num_dist_word_selected = num_dist_word_selected
        self.temperature_initial = temperature_initial
        self.batch_first = batch_first

    def vocab_dist(self, context_q, context_c, decoded):
        """
        distribution over the vocabulary
        :param context_q: please see forward()
        :param context_c: please see forward()
        :param decoded: please see forward()
        :return: probabilities, [bsz, s_len, vocab_size]
        """
        concatenated = torch.cat([context_q, context_c, decoded], dim=2)
        logits = nn.Linear(concatenated.shape[2], len(self.idx2word), bias=True)(concatenated)  # [bsz, s_len, vocab_size]
        return nn.Softmax(dim=2)(logits)

    def latent_representation(self, context_q, context_c, decoded, cq):
        """
        :param context_q: please see forward()
        :param context_c: please see forward()
        :param decoded: please see forward()
        :param cq: please see forward()
        :return: latent representation h, [bsz, s_len, latent_dim]
        :return: v, [bsz, s_len, 2xencoder_hidden_dim + 2xencoder_hidden_dim + decoder_hidden_dim + embedding_dim]
        :return: h_dist, distribution of latent representation
        """
        v = torch.cat([context_q, context_c, decoded, cq], dim=2)
        mu = torch.tanh(nn.Linear(v.shape[2], self.latent_dim, bias=False)(v))
        sigma = torch.exp(torch.tanh(nn.Linear(v.shape[2], self.latent_dim, bias=False)(v))) # todo : non-stability catastrophe
        variance = torch.diag(sigma ** 2)
        if self.mode == 'gaussian':
            h_dist = torch.distributions.MultivariateNormal(mu, variance)
            # todo : this epsilon is changed but not debugged yet, please pay attention
            epsilon = torch.distributions.MultivariateNormal(torch.zeros(mu.shape), torch.eye(mu.shape[2]).expand([mu.shape[0], mu.shape[1], mu.shape[2], mu.shape[2]])).sample()
        # todo : and attention please, in training phase, we need Q_phi as distribution, so this distribution is for testing phase, because we have KL-divergence of both two in the loss, so this distribution is also optimized, don't worry.
        # if self.mode == 'gaussian':
        #     # return torch.distributions.MultivariateNormal(mu, torch.pow(sigma, 2)).sample()
        #     # epsilon = torch.distributions.MultivariateNormal(torch.zeros_like(mu), torch.ones_like(sigma)).sample()
        #     stack = []
        #     for i in range(mu.shape[0]):
        #         sub_stack = []
        #         for j in range(mu.shape[1]):
        #             sub_stack.append(torch.distributions.MultivariateNormal(torch.zeros(mu.shape[2]), torch.eye(mu.shape[2])).sample())
        #         stack.append(torch.stack(sub_stack, dim=0))
        #     epsilon = torch.stack(stack, dim=0)
            #  todo : we can use this reparameterization trick for variational Bayesian methods here ? or rather just in the loss (lower bound) ?

            # very perspicuous reparameterization trick to solve auto-gradient problem of sampling : convert N(mu, sigma**2).sample to mu+sigma*epsilon, which is also expectation
            return mu + sigma * epsilon, v, h_dist


    def latent_variable(self, h, num_step):
        """
        :param h: latent representation, [bsz, s_len, latent_dim]
        :param num_step: please see forward()
        :return: estimator (one-hot) for latent variable, [bsz, s_len, num_dist_word_selected]
        """
        logits = nn.Linear(h.shape[2], self.num_dist_word_selected)(h)
        pi = nn.Softmax(dim=2)(logits)
        # todo : can we treat this by batch
        gumbel = torch.distributions.gumbel.Gumbel(torch.tensor([[[0.0 for k in range(pi.shape[2])]
                                                                  for j in range(pi.shape[1])]
                                                                 for i in range(pi.shape[0])]),
                                                   torch.tensor([[[1.0 for k in range(pi.shape[2])]
                                                                  for j in range(pi.shape[1])]
                                                                 for i in range(pi.shape[0])])).sample()
        gplogpi = gumbel + torch.log(pi)
        # with temperature approaches zero, estimator becomes an one-hot vector (Gumbel-Max trick).
        estimator = nn.Softmax(dim=2)(gplogpi/(self.temperature_initial/num_step))
        return estimator


    def word_selection(self, att_c, att_q, vocab_dist, estimator, examples):
        """
        - we do not use the piece-wise function, but multiplication of the choices and an one-hot vector.
        - this works since we have convert the discrete latent variable into continuous one-hot vector by Gumbel-Max trick.
        :param att_c: please see forward()
        :param att_q: please see forward()
        :param vocab_dist: probs_from_vocab, [bsz, s_len, vocab_size]
        :param estimator: estimator (one-hot) for latent variable, [bsz, s_len, num_dist_word_selected]
        :param examples: please see forward()
        :return: probs, [bsz, s_len, vocab_size] # todo : in the formula, we get probs for word t+1 through information from word t (t: index of answer), but this realization is not t+1 but t.
        """
        assert(self.num_dist_word_selected == 3)
        splitted_att_c = torch.split(att_c, 1, dim=0)
        splitted_att_q = torch.split(att_q, 1, dim=0)
        splitted_vocab_dist = torch.split(vocab_dist, 1, dim=0)
        splitted_estimator = torch.split(estimator, 1, dim=0)

        # for each in batch
        # todo : can we enumerate together : for i, each_att_c, each_att_q, each_vocab_dist in enumerate(...) ?
        # todo : or this can be done by batch, i.e. we don't need for loop ! text_to_idx is the only function that needs to split batch.
        stack = []
        for i in range(len(splitted_att_c)):

            # splitted_att_c[i]: [1, en_len, s_len]
            list_idx = text_to_idx(examples[i]['case'], self.word2idx)
            map_idx = torch.tensor(idx_to_onehot(list_idx, dim=len(self.idx2word)), dtype=torch.float)  # [en_len, vocab_size]
            transpoded_splitted_att_c = torch.transpose(torch.squeeze(splitted_att_c[i]), 0, 1)  # [s_len, en_len] todo : original att_c not changed ?
            probs_from_cases = torch.matmul(transpoded_splitted_att_c, map_idx)  # [s_len, vocab_size]

            # splitted_att_q[i]: [1, en_len, s_len]
            list_idx = text_to_idx(examples[i]['question'], self.word2idx)
            map_idx = torch.tensor(idx_to_onehot(list_idx, dim=len(self.idx2word)), dtype=torch.float)  # [en_len, vocab_size]
            transpoded_splitted_att_q = torch.transpose(
                torch.squeeze(splitted_att_q[i]), 0, 1)  # [s_len, en_len]
            probs_from_questions = torch.matmul(transpoded_splitted_att_q, map_idx)  # [s_len, vocab_size]

            probs_from_vocab_dist = torch.squeeze(splitted_vocab_dist[i])  # [s_len, vocab_size]

            this_estimator = torch.squeeze(splitted_estimator[i])  # [s_len, num_dist_word_selected]=[s_len, 3]

            stacked = torch.stack([probs_from_questions, probs_from_cases, probs_from_vocab_dist], dim=2)

            stack.append(torch.squeeze(torch.bmm(stacked, this_estimator.unsqueeze(dim=2))))  # [s_len, vocab_size]
        return torch.stack(stack, dim=0)  # [bsz, s_len, vocab_size]


    def forward(self, cq, att_c, att_q, context_c, context_q, decoded, examples, num_step):
        """
        :param cq : concatenation of cases and questions (vector form, NOT encoded form), todo : may be replaced by answer in training phase
        :param att_c: attention map for cases, [bsz, c_len, s_len] where s_len is decoded length
        :param att_q: attention map for questions, [bsz, q_len, s_len] where s_len is decoded length
        :param context_c: context vector for cases, [bsz, s_len, 2xencoder_hidden_dim]
        :param context_q: context vector for questions, [bsz, s_len, 2xencoder_hidden_dim]
        :param decoded: decoded state.
        :param examples: this batch of original tokenized cases and questions in str.
        :param num_step: which traning step.
        :return: probs: [bsz, s_len, vocab_size], then we can use this to generate answer! (either in a greedy way or in a beam search way)
        :return: h: [bsz, s_len, latent_dim]
        :return: v: [bsz, s_len, 2xencoder_hidden_dim + 2xencoder_hidden_dim + decoder_hidden_dim + embedding_dim]
        :return: h_dist, distribution of latent representation
        """
        vocab_dist = self.vocab_dist(context_q, context_c, decoded)
        h, v, h_dist = self.latent_representation(context_q, context_c, decoded, cq)
        estimator = self.latent_variable(h, num_step)
        probs = self.word_selection(att_c, att_q, vocab_dist, estimator, examples)
        return probs, h, v, h_dist, att_c, att_q, vocab_dist


