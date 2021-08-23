from config import latentqa_cail2021 as config
import torch
from data.dataloader import get_train_loader
import preprocess.vocab as vocab
from model.inference import LatentQA
import tensorboardX
import torch.nn as nn
from preprocess.vocab import text_to_idx, idx_to_onehot
from preprocess.text_utils import BOS
from torch.distributions import MultivariateNormal, kl_divergence


DEVICE = torch.device('cuda') if config.train_cfg['use_gpu'] is True and torch.cuda.is_available() else 'cpu'


class LowerBoundLoss(nn.Module):
    def __init__(self):
        super(LowerBoundLoss, self).__init__

    def dist_qhvw(self, h, v, onehot):
        r = nn.Linear(onehot.shape[2], h.shape[2], bias=False)(onehot)  # todo : what is the dimension of r ?
        print(r.shape)  # [bsz, s_len, latent_dim]
        vr = torch.cat([v, r], dim=2)
        mu = torch.tanh(nn.Linear(vr.shape[2], h.shape[2], bias=False)(vr))  # [bsz, s_len, latent_dim]
        sigma = torch.exp(torch.tanh(nn.Linear(vr.shape[2], h.shape[2], bias=False)(vr)))  # [bsz, s_len, latent_dim]
        variance = torch.diag(sigma ** 2)
        return mu, sigma, MultivariateNormal(mu, variance)

    def expectation(self, mu, sigma, onehot, att_c, att_q, vocab_dist, num_step, examples):
        """
        for the training phase, the similar one in selector is for testing phase. the difference is : we change distribution for latent representation.
        :param mu:
        :param sigma:
        :param onehot: [bsz, s_len, vocab_size]
        :return:
        """
        epsilon = MultivariateNormal(torch.zeros(mu.shape), torch.eye(mu.shape[2]).expand(
            [mu.shape[0], mu.shape[1], mu.shape[2], mu.shape[2]])).sample()  # [bsz, s_len, latent_dim]
        latent = mu + sigma * epsilon  # [bsz, s_len, latent_dim]
        logits = nn.Linear(latent.shape[2], config.model_cfg['num_dist_word_selected'])(latent)
        pis = nn.Softmax(dim=2)(logits)
        # todo : can we treat this by batch
        gumbel = torch.distributions.gumbel.Gumbel(torch.tensor([[[0.0 for k in range(pis.shape[2])]
                                                                  for j in range(pis.shape[1])]
                                                                 for i in range(pis.shape[0])]),
                                                   torch.tensor([[[1.0 for k in range(pis.shape[2])]
                                                                  for j in range(pis.shape[1])]
                                                                 for i in range(pis.shape[0])])).sample()
        gplogpi = gumbel + torch.log(pis)
        # with temperature approaches zero, estimator becomes an one-hot vector (Gumbel-Max trick).
        estimator = nn.Softmax(dim=2)(gplogpi / (self.temperature_initial / num_step))

        assert (config.model_cfg['num_dist_word_selected'] == 3)
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
            map_idx = torch.tensor(idx_to_onehot(list_idx, dim=len(self.idx2word)),
                                   dtype=torch.float)  # [en_len, vocab_size]
            transpoded_splitted_att_c = torch.transpose(torch.squeeze(splitted_att_c[i]), 0,
                                                        1)  # [s_len, en_len] todo : original att_c not changed ?
            probs_from_cases = torch.matmul(transpoded_splitted_att_c, map_idx)  # [s_len, vocab_size]

            # splitted_att_q[i]: [1, en_len, s_len]
            list_idx = text_to_idx(examples[i]['question'], self.word2idx)
            map_idx = torch.tensor(idx_to_onehot(list_idx, dim=len(self.idx2word)),
                                   dtype=torch.float)  # [en_len, vocab_size]
            transpoded_splitted_att_q = torch.transpose(
                torch.squeeze(splitted_att_q[i]), 0, 1)  # [s_len, en_len]
            probs_from_questions = torch.matmul(transpoded_splitted_att_q, map_idx)  # [s_len, vocab_size]

            probs_from_vocab_dist = torch.squeeze(splitted_vocab_dist[i])  # [s_len, vocab_size]

            this_estimator = torch.squeeze(splitted_estimator[i])  # [s_len, num_dist_word_selected]=[s_len, 3]

            stacked = torch.stack([probs_from_questions, probs_from_cases, probs_from_vocab_dist], dim=2)

            stack.append(torch.squeeze(torch.bmm(stacked, this_estimator.unsqueeze(dim=2))))  # [s_len, vocab_size]
        probs = torch.stack(stack, dim=0)  # [bsz, s_len, vocab_size]

        # todo : w -> w', which is one time-step in advance ? (i.e. w_t -> w_{t+1})
        slen_slen = torch.bmm(probs, torch.transpose(onehot, 1, 2))
        p_theta = torch.stack([torch.diag(slen_slen[i]) for i in range(probs.shape[0])], dim=0)  # [bsz, s_len]
        return torch.log(p_theta)  # [bsz, s_len]

    def forward(self, h, v, onehot, h_dist, att_c, att_q, vocab_dist, num_step, examples):
        """
        lower bound loss (batch-wise and length-wise mean)
        :param probs: [bsz, s_len, vocab_size]
        :param h: [bsz, s_len, latent_dim]
        :param v: [bsz, s_len, 2xencoder_hidden_dim]
        :param onehot form of examples, [bsz, s_len, vocab_size]
        :return: loss
        """
        print(onehot)
        mu, sigma, dist_qhvw = self.dist_qhvw(h, v, onehot, h_dist)
        dist_phv = h_dist
        kl_term = kl_divergence(dist_qhvw, dist_phv)  # [bsz, s_len]
        expectation = self.expectation(mu, sigma, onehot, att_c, att_q, vocab_dist, num_step, examples)  # [bsz, s_len]
        loss_unmerged = expectation - kl_term
        return torch.mean(torch.mean(loss_unmerged, dim=1), dim=0)


def train(model_cfg, dataloader, idx2word, word2idx, num_epochs, batch_first=True):
    print('training...')
    model = LatentQA(model_cfg, idx2word, word2idx, batch_first)
    criterion = LowerBoundLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15, initial_accumulator_value=0.1)
    print('cuda is available ? ' + str(torch.cuda.is_available()))
    model.to(DEVICE)
    model.train()
    for epoch in range(num_epochs):
        for num_step, (cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples) in enumerate(dataloader):
            # :param examples: [{'case': xxx, 'question': xxx}, ...], and case/question : ['第', '一句', '话']
            # todo : something goes wrong : for bidirectional lstms, we need calculate a forward loss and backward loss, and then loss = (forw_loss + backw_loss) / 2
            # this probs is for testing phase, we calculate the probs for training phase in LowerBoundLoss
            # todo : we should optimize the code separation of training and testing phase, by a mode parameter (if mode == 'train', dist=QHVW else dist=HV)
            probs, h, v, h_dist, att_c, att_q, vocab_dist = model(cases, questions, lengths_case, lengths_question, cases_pad_mask, questions_pad_mask, examples, num_step)
            onehot = [idx_to_onehot(text_to_idx(item['case'] + item['question'], word2idx)) for item in examples]  # todo : change to answer
            print(probs.shape)
            loss = criterion(h, v, onehot, h_dist, att_c, att_q, vocab_dist, num_step, examples)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss@step-%d@epoch-@d : %f' % (num_step, epoch, loss.item()))
            # todo : visualization !
            # todo : evaluation (aka, testing)

def main(train_cfg, model_cfg, *args, **kwargs):
    if train_cfg['use_char'] is False:
        idx2word = vocab.idx2word(train_cfg['vocab_word_path'], train_cfg['vocab_size'])
        word2idx = vocab.word2idx(train_cfg['vocab_word_path'], train_cfg['vocab_size'])
    else:
        idx2word = vocab.idx2word(train_cfg['vocab_char_path'], train_cfg['vocab_size'])
        word2idx = vocab.word2idx(train_cfg['vocab_char_path'], train_cfg['vocab_size'])

    dataloader = get_train_loader(train_cfg)
    train(model_cfg, dataloader, idx2word, word2idx, train_cfg['num_epochs'], train_cfg['batch_first'])


if __name__ == '__main__':
    # parse command config, c.f. [argparse](https://zhuanlan.zhihu.com/p/56922793)
    main(config.train_cfg, config.model_cfg)

