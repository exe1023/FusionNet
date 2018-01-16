import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

use_cuda = torch.cuda.is_available()

class FullyAttention(nn.Module):
    '''
    Attention proposed in section 2.3
    '''
    def __init__(self, in_dim, attn_dim):
        super(FullyAttention, self).__init__()
        self.U = nn.Linear(in_dim, attn_dim)
        
        self.D = nn.Parameter(torch.Tensor(attn_dim))
        stdv = 1. / math.sqrt(self.D.size(0))
        self.D.data.uniform_(-stdv, stdv)
    
    def forward(self, h_output, h_context, context):
        '''
        Fusing output to context
        Args:
            h_output: (batch, out_len, in_dim)
            h_context: (batch, con_len, in_dim)
            context: (batch, con_len, hidden_dim)
        '''
        batch = h_output.size(0)
        in_dim = h_output.size(2)
        con_len = h_context.size(1)

        # (batch, len, in_dim) -> (batch, len, attn_dim)
        o_feature = F.relu(self.U(h_output))
        c_feature = F.relu(self.U(h_context))
        # (batch, out_len, attn_dim) * (attn_dim) -> (batch, out_len, attn_dim)
        o_feature *= self.D
        # (batch, out_len, attn_dim) * (batch, con_len, attn_dim) -> (batch, out_len, con_len)
        attn = torch.bmm(o_feature, c_feature.transpose(1, 2))
        attn = F.softmax(attn.view(-1, con_len), -1).view(batch, -1, con_len)
        # (batch, out_len, con_len) * (batch, con_len, hidden_dim) -> (batch, out_len, hidden_dim)
        mix = torch.bmm(attn, context)
        return mix

class WordAttention(nn.Module):
    '''
    Word-level attention
    '''
    def __init__(self, dim):
        '''
        Args:
            dim: input_dim = hidden_size = output_dim
        '''
        super(WordAttention, self).__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, c, q):
        '''
        Args:
            c: (batch, c_len, dim)
            q: (batch, q_len, dim)
        Output:
            mix: (batch, c_len, dim)
        '''
        batch = c.size(0)
        word_dim = c.size(2)
        q_len = q.size(1)

        c_feature = F.relu(self.linear(c))
        q_feature = F.relu(self.linear(q))
        # (batch, c_len, dim) * (batch, q_len, dim) -> (batch, c_len, q_len)
        attn = torch.bmm(c_feature, q_feature.transpose(1, 2))
        attn = F.softmax(attn.view(-1, q_len), -1).view(batch, -1, q_len)
        # (batch, c_len, q_len) * (batch, q_len, dim) -> (batch, c_len, dim)
        mix = torch.bmm(attn, q)
        return mix

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    def forward(self, input):
        return self.embedding(input)

    def init_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

class DocReader(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers,
                 bidirectional=True,
                 dropout = 0.6,
                 rnn_type='lstm'):
        super(DocReader, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout)
        elif rnn_type =='lstm':
            self.rnn = nn.LSTM(input_size, 
                               hidden_size, 
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True,
                               dropout=dropout)
        else:
            print('Unexpected rnn type')
            exit()
    
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        bidirectional = 2 if self.bidirectional else 1
        h = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))

        if self.rnn_type == 'gru':
            return h.cuda() if use_cuda else h
        else:
            c = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))
            return (h.cuda(), c.cuda()) if use_cuda else (h, c)

def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).
    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

class PointerNet(nn.Module):
    def __init__(self, 
                 context_size,
                 question_size,
                 hidden_size=250
                 ):
        super(PointerNet, self).__init__()
        self.fc_q = nn.Linear(question_size, 1)

        self.fc_start = nn.Linear(question_size, context_size)
        self.fc_end = nn.Linear(question_size, context_size)
        
        self.start2end = nn.GRU(context_size, 
                                question_size,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True)
    def forward(self, context, question):
        batch = question.size(0)
        q_len = question.size(1)
        q_dim = question.size(2)

        q_flat = question.contiguous().view(-1, q_dim)
        scores = self.fc_q(question.view(-1, q_dim))
        #scores.data.masked_fill_(x_mask.data, -float('inf'))
        beta = F.softmax(scores, -1).view(batch, q_len)
        u_q = weighted_avg(question, beta)

        start_weight = self.fc_start(u_q)
        # (batch, c_len, c_dim) * (batch, c_dim, 1) -> (batch, c_len, 1)
        start_attn = context.bmm(start_weight.unsqueeze(2)).squeeze(2)
        # start_attn.data.masked_fill_(x_mask.data, -float('inf'))
        
        start = F.softmax(start_attn, -1)
        
        v_q, _ = self.start2end(weighted_avg(context, start).unsqueeze(1),
                                u_q.unsqueeze(0))
        end_weight = self.fc_end(v_q.squeeze(1))
        end_attn = context.bmm(end_weight.unsqueeze(2)).squeeze(2)
        
        end = F.softmax(end_attn, -1)

        return start, end, start_attn, end_attn


class FusionNet(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_dim=300,
                 hidden_size=125,
                 rnn_layer=1,
                 dropout=0.4,
                 pretrained_embedding=None
                 ):
        super(FusionNet, self).__init__()
        self.embedding = Embedding(vocab_size, word_dim)
        if pretrained_embedding is not None:
            self.embedding.init_embedding(pretrained_embedding)
        
        # --- FusionNet RNN reader --- #
        # low(high)-level concepts
        q_dim = word_dim
        c_dim = word_dim + word_dim + 1
        self.q_low_reader = DocReader(input_size=q_dim, 
                                  hidden_size=hidden_size,
                                  num_layers=rnn_layer,
                                  bidirectional=True,
                                  dropout=dropout,
                                  rnn_type='lstm')
        self.q_high_reader = DocReader(input_size=hidden_size * 2,
                                       hidden_size=hidden_size,
                                       num_layers=rnn_layer,
                                       bidirectional=True,
                                       dropout=dropout,
                                       rnn_type='lstm')
        self.c_low_reader = DocReader(input_size=c_dim,
                                  hidden_size=hidden_size,
                                  num_layers=rnn_layer,
                                  bidirectional=True,
                                  dropout=dropout,
                                  rnn_type='lstm')
        self.c_high_reader = DocReader(input_size=hidden_size * 2,
                                    hidden_size=hidden_size,
                                    num_layers=rnn_layer,
                                    bidirectional=True,
                                    dropout=dropout,
                                    rnn_type='lstm')
        # question understanding
        self.qu_reader = DocReader(input_size=(hidden_size * 2) * 2,
                                   hidden_size=hidden_size,
                                   num_layers=rnn_layer,
                                   bidirectional=True,
                                   dropout=dropout,
                                   rnn_type='lstm')
        # Fusion reader
        fully_fused_dim = (hidden_size * 2) * 5
        self.fused_reader = DocReader(input_size=fully_fused_dim,
                                      hidden_size=hidden_size,
                                      num_layers=rnn_layer,
                                      bidirectional=True,
                                      dropout=dropout,
                                      rnn_type='lstm')
        # final reader
        self_fused_dim = (hidden_size * 2) * 2
        self.cu_reader = DocReader(input_size=self_fused_dim,
                                           hidden_size=hidden_size,
                                           num_layers=rnn_layer,
                                           bidirectional=True,
                                           dropout=dropout,
                                           rnn_type='lstm')
        # --- Attention --- #
        self.word_attention = WordAttention(word_dim)
        # fusion between context and quesiotn
        h_dim = word_dim + (hidden_size * 2) * 2
        attn_dim = hidden_size * 2
        self.low_fustion = FullyAttention(h_dim, attn_dim)
        self.high_fustion = FullyAttention(h_dim, attn_dim)
        self.understand_fustion = FullyAttention(h_dim, attn_dim)
        # self-boosted fusion
        h_dim = word_dim + (hidden_size * 2) * 6
        self.self_fusion = FullyAttention(h_dim, attn_dim)

        # --- Pointer Network --- #
        self.pointer_net = PointerNet(hidden_size*2,
                                      hidden_size*2,
                                      )

        # Dropout layer
        self.dropout_emb = nn.Dropout(p=0.3)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, context, question, appear):
        '''
        Args:
            context:  (batch, c_len)
            question: (batch, q_len)
            appear: (batch, c_len)
        '''
        batch = context.size(0)
        c_len = context.size(1)
        q_len = context.size(1)
        # Embed word
        c_word = self.embedding(context)
        q_word = self.embedding(question)
        
        c_word = self.dropout_emb(c_word)
        q_word = self.dropout_emb(q_word)
        # word_attn: (batch, c_len, word_dim)
        word_attn = self.word_attention(c_word, q_word)
        ''' 
        TODO:
        c_feature, q_feature:
            contextualized vector
        c_feature: 
            POS, NER, Normalized term frequency
        '''
        c_feature = torch.cat((c_word, appear.view(batch, c_len, 1), word_attn), 2)
        q_feature = q_word
        # Get low(high)-level concepts
        # c_low: (batch, c_len, hidden_size * 2)
        # q_low: (batch, q_len, hidden_size * 2)
        c_low, _ = self.c_low_reader(c_feature, self.c_low_reader.init_hidden(batch))
        q_low, _ = self.q_low_reader(q_feature, self.q_low_reader.init_hidden(batch))
        
        c_low = self.dropout(c_low)
        q_low = self.dropout(q_low)

        c_high, _ = self.c_high_reader(c_low, self.c_high_reader.init_hidden(batch))
        q_high, _ = self.q_high_reader(q_low, self.q_high_reader.init_hidden(batch))

        c_high = self.dropout(c_high)
        q_high = self.dropout(q_high)
        # Question Understanding
        qu, _ = self.qu_reader(torch.cat((q_low, q_high), 2),
                            self.qu_reader.init_hidden(batch))
        
        qu = self.dropout(qu)
        # Form history of word
        # TODO: contextualized vector
        c_history = torch.cat((c_word, c_low, c_high), 2)
        q_history = torch.cat((q_word, q_low, q_high), 2)
        # Low, High, Understanding fusion
        low_fusion = self.low_fustion(c_history, q_history, q_low)
        high_fusion = self.high_fustion(c_history, q_history, q_high)
        understand_fusion = self.understand_fustion(c_history, q_history, qu)
        # Read fully-fused informaiotn
        fused_v, _ = self.fused_reader(torch.cat((c_low, c_high, low_fusion, high_fusion, understand_fusion), 2),
                                    self.fused_reader.init_hidden(batch))
        
        fused_v = self.dropout(fused_v)
        # self-boosted fusion
        c_history = torch.cat((c_history, low_fusion, high_fusion,
                                understand_fusion, fused_v), 2)
        
        self_fusion = self.self_fusion(c_history, c_history, fused_v)
        cu, _ = self.cu_reader(torch.cat((fused_v, self_fusion), 2),
                            self.cu_reader.init_hidden(batch))
        
        cu = self.dropout(cu)
        # --- Pointer Network --- #
        start, end, start_attn, end_attn = self.pointer_net(cu, qu)
        return start, end, start_attn, end_attn

def decode(score_s, score_e, top_n=1, max_len=None):
    """Take argmax of constrained score_s * score_e.
    Args:
        score_s: independent start predictions
        score_e: independent end predictions
        top_n: number of top scored pairs to take
        max_len: max span length to consider
    """
    pred_s = []
    pred_e = []
    pred_score = []
    max_len = max_len or score_s.size(1)
    for i in range(score_s.size(0)):
        # Outer product of scores to get full p_s * p_e matrix
        scores = torch.ger(score_s[i], score_e[i])

        # Zero out negative length and over-length span scores
        scores.triu_().tril_(max_len - 1)

        # Take argmax or top n
        scores = scores.numpy()
        scores_flat = scores.flatten()
        if top_n == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < top_n:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, top_n)[0:top_n]
            idx_sort = idx[np.argsort(-scores_flat[idx])]
        s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
        pred_s.append(s_idx)
        pred_e.append(e_idx)
        pred_score.append(scores_flat[idx_sort])
    return pred_s, pred_e, pred_score


if __name__ == '__main__':
    fusion_net = FusionNet(300)
    context = Variable(torch.rand(32, 100).type(torch.LongTensor))
    question = Variable(torch.rand(32, 20).type(torch.LongTensor))
    if use_cuda:
        fusion_net = fusion_net.cuda()
        context = context.cuda()
        question = question.cuda()
    start, end, start_attn, end_attn = fusion_net(context, question)
    start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1)
    print(start)
    print(end)
    print(scores)



