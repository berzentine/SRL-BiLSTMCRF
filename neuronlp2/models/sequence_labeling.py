__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import ChainCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM
from ..nn import Embedding
from ..nn import utils


class BiRecurrentConv(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, trig_dim, num_trigs, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiRecurrentConv, self).__init__()

        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        #TODO: self.trigger_embedd = Embedding(??, trigger_dim, init_embedding=embedd_trigger)
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char)
        self.trig_embedd = Embedding(num_trigs, trig_dim, init_embedding=None)
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_rnn = nn.Dropout(p_rnn)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        # TODO Change here since the word dim + trig dim RNN should be
        self.rnn = RNN(word_dim + num_filters + trig_dim, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=p_rnn)

        self.dense = None
        out_dim = hidden_size * 2
        self.od = out_dim
        if tag_space:
            self.dense = nn.Linear(out_dim, tag_space)
            out_dim = tag_space
            self.od = out_dim
        self.dense_softmax = nn.Linear(out_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

    def _get_rnn_output(self, input_word, input_char, input_trig, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()

        # [batch, length, word_dim]
        word = self.word_embedd(input_word)
        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        trig = self.trig_embedd(input_trig)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char, trig], dim=2)
        # apply dropout
        input = self.dropout_in(input)
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            seq_output, hn = self.rnn(seq_input, hx=hx)
            # TODO Nidhi: need to change here for concatenating the rnn output with the trigger embeddings
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size]
            output, hn = self.rnn(input, hx=hx)
        output = self.dropout_rnn(output)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = F.elu(self.dense(output))

        return output, hn, mask, length


    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        return output, mask, length

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # [batch, length, tag_space]
        output, mask, length = self.forward(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_labels]
        output = self.dense_softmax(output)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)

        if length is not None and target.size(1) != mask.size(1):
            max_len = length.max()
            target = target[:, :max_len].contiguous()

        if mask is not None:
            return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / mask.sum(), \
                   (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)).sum() / num, \
                   (torch.eq(preds, target).type_as(output)).sum(), preds


class BiVarRecurrentConv(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5):
        super(BiVarRecurrentConv, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn)

        self.dropout_in = None
        self.dropout_rnn = nn.Dropout2d(p_rnn)

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim + num_filters, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True, dropout=(p_in, p_rnn))

    def _get_rnn_output(self, input_word, input_char, mask=None, length=None, hx=None):
        # [batch, length, word_dim]
        word = self.word_embedd(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char], dim=2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask, hx=hx)
        # apply dropout for the output of rnn
        output = self.dropout_rnn(output.transpose(1, 2)).transpose(1, 2)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = F.elu(self.dense(output))

        return output, hn, mask, length


class BiRecurrentConvCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, trig_dim, num_trigs, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, bigram=False):
        super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, trig_dim, num_trigs, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        print 'here'
        output, _, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, input_trig, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_trig, mask=mask, length=length, hx=hx)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word, input_char, input_trig, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, input_trig, mask=mask, length=length, hx=hx)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()
