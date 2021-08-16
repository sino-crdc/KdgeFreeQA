import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, embedding_dim, encoder_dim, num_encoder_layers=1, batch_first=True):
        """
        question encoder and case encoder
        :param embedding_dim: vector dimension.
        :param encoder_dim: encoder hidden dimension.
        :param num_encoder_layers: number of encoder layers.
        :param batch_first: if batch_size is the first dimension.
        """
        # case lstm encoder
        super(Encoder, self).__init__()
        self.clstm = nn.LSTM(input_size=embedding_dim, hidden_size=encoder_dim, num_layers=num_encoder_layers,
                             batch_first=batch_first, bidirectional=True)
        # question lstm encoder
        self.qlstm = nn.LSTM(input_size=embedding_dim, hidden_size=encoder_dim, num_layers=num_encoder_layers,
                             batch_first=batch_first, bidirectional=True)

    def forward(self, cases, questions, lengths_case=None, lengths_question=None, cases_pad_mask=None, questions_pad_mask=None):
        """
        input encoder.
        :param cases: cases in vector form.
        :param questions: questions in vector form.
        :param lengths_case
        :param lengths_question
        :param cases_pad_mask
        :param questions_pad_mask
        :return: encoded cases and encoded questions. both in [bsz, en_len, 2xen_dim]
        """
        # we returns only outputs, don't returns (hn, cn)
        return self.clstm(cases)[0], self.qlstm(questions)[0]  # output_dim= [bsz, length, 2 x hidden_dim]


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_decoder_layers=1, batch_first=True):
        """
        s decoder
        :param input_dim: encoder hidden dimension for cases, don't forget to x2, i.e. 2 x en_dim because of bidirection.
        :param hidden_dim: decoder hidden dimension
        :param num_decoder_layers: number of decoder layers
        :param batch_first
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_decoder_layers,
                            batch_first=batch_first, bidirectional=False)

    def forward(self, cases, questions, lengths_case=None, lengths_question=None, cases_pad_mask=None, questions_pad_mask=None):
        """
        answer decoder. If there is no available answer, we use length-wise concatenation of [cases, questions].
        :param cases: cases in encoded form.
        :param questions: questions in encoded form.
        :param lengths_case
        :param lengths_question
        :param cases_pad_mask
        :param questions_pad_mask
        :return: decoder state. [bsz, de_len, de_dim]
        """
        # todo : in training phase, we should replace [cases, questions] by answer, i.e., the decoder encodes the answer sequence in traning phase while encodes [cases, questions] in test phase
        # we returns only outputs, don't returns (hn, cn)
        return self.lstm(torch.cat([cases, questions], dim=1))[0]


class AttentionLayer(nn.Module):
    def __init__(self, c_enc_dim, q_enc_dim, dec_dim, batch_first=True):
        """
        two attention maps
        :param c_enc_dim: encoder hidden dimension for cases, don't forget to x2, i.e. 2 x en_dim because of bidirection.
        :param q_enc_dim: encoder hidden dimension for questions, don't forget to x2, i.e. 2 x en_dim because of bidirection.
        :param dec_dim: decoder hidden dimension
        :param batch_first
        """
        super(AttentionLayer, self).__init__()
        self.batch_first = batch_first
        self.wc = nn.Linear(c_enc_dim, dec_dim, bias=False)
        self.uc = nn.Linear(dec_dim, dec_dim, bias=True)
        self.wq = nn.Linear(q_enc_dim, dec_dim, bias=False)
        self.uq = nn.Linear(dec_dim, dec_dim, bias=True)
        # the remaining learnable matrix are not defined in __init__ since there is a loop.

    def forward(self, cases, questions, decoded):
        """
        attention map for cases and questions.
        :param cases: cases in encoded form.
        :param questions: questions in encoded form.
        :param decoded: decoder state.
        :param batch_first
        :return: att_c, att_q, both [bsz, en_len, de_len]
        """
        # todo : for case, there is a coverage mechanism to avoid generate repetitive text.
        cases = self.wc(cases)  # this does not change the hidden dimension of parameter cases, i.e. beyond this function, hidden_dim of cases is still x2
        decoded = self.uc(decoded)
        questions = self.wq(questions)
        decoded = self.uc(decoded)
        decodeds = torch.split(decoded, 1, dim=1)
        stack_c = []
        stack_q = []
        for each in decodeds:
            dec_c, _ = torch.broadcast_tensors(each, cases)
            dec_q, _ = torch.broadcast_tensors(each, questions)
            activated_c = torch.tanh(dec_c + cases)
            activated_q = torch.tanh(dec_q + questions)
            logits_c = nn.Linear(decoded.shape[2], 1, bias=False)(activated_c).squeeze()
            logits_q = nn.Linear(decoded.shape[2], 1, bias=False)(activated_q).squeeze()
            at_c = nn.Softmax(dim=1)(logits_c)
            at_q = nn.Softmax(dim=1)(logits_q)
            stack_c.append(at_c)
            stack_q.append(at_q)
        att_c = torch.stack(stack_c, dim=2)
        att_q = torch.stack(stack_q, dim=2)
        return att_c, att_q

        # for each in decodeds:
        #     dec, _ = torch.broadcast_tensors(each, questions)
        #     activated = torch.tanh(dec + questions)
        #     logits = nn.Linear(decoded.shape[2], 1, bias=False)(activated).squeeze
        #     at = nn.Softmax(dim=1)(logits)
        #     stack.append(at)
        # att_c = torch.stack(stack, dim=2)


def context_vector(att, encoded):
    """
    context vector.
    :param att: attention map for the encoded, [bsz, en_len, de_len]
    :param encoded: encoded state of input, [bsz, en_len, 2xen_dim] (because there is a stack of both forward and backward lstms, dim = en_hidden dim x2)
    :param batch_first
    :return: context vector, [bsz, de_len, en_dim]
    """
    return torch.bmm(torch.transpose(att, 1, 2), encoded)  # att shape not changed.


if __name__ == '__main__':
    """
    the dimensional analysis is very difficult, but this main function may help you.
    """
    # data construction
    array = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[11, 22, 33, 44], [55, 66, 77, 88]], [[1, 2, 3, 4], [5, 6, 7, 8]],
             [[11, 22, 33, 44], [55, 66, 77, 88]]]
    data1 = torch.tensor(array, dtype=torch.float32)  # [4, 2, 4]
    data2 = torch.tensor(array, dtype=torch.float32)  # [4, 2, 4]

    # layers: from encoder to context vector
    encoder = Encoder(4, 10, 1, True)
    cases, questions = encoder(data1, data2)  # cases = [4, 2, 20]
    decoder = Decoder(20, 10, 1, True)
    decoded = decoder(cases, questions)   # [4, 4, 10]

    # cases = nn.Linear(20, 10, bias=False)(cases)  # [4, 2, 10]
    # decoded = nn.Linear(10, 10, bias=True)(decoded)  # [4, 4, 10]
    # decodeds = torch.split(decoded, 1, dim=1)  # [4, 1, 10], [4, 1, 10], [4, 1, 10], [4, 1, 10]
    # print(decodeds[0].shape)
    # stack = []
    # for each in decodeds:
    #     dec, _ = torch.broadcast_tensors(each, cases)  # [4, 2, 10]
    #     print(decodeds[0].shape)
    #     print(dec.shape)
    #     activated = torch.tanh(dec + cases)
    #     logis = nn.Linear(10, 1, bias=False)(activated).squeeze()   # [4, 2]
    #     # print(decoded.shape[2])
    #     at = nn.Softmax(dim=1)(logis)
    #     stack.append(at)
    # print(decoded[0].shape)
    # att = torch.stack(stack, dim=2)
    # print(att.shape)  # [4, 2, 4]

    attention_layer = AttentionLayer(20, 20, 10)
    att_c, att_q = attention_layer(cases, questions, decoded)
    print(att_c.shape)
    # print(att_q.shape)
    context_c = context_vector(att_c, cases)
    print(context_c.shape)
    context_q = context_vector(att_q, questions)
    print(context_q.shape)



























