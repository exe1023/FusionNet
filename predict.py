import argparse
import data
from module import FusionNet, decode
from metrics import batch_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch import optim

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Model and Training parameters')
# Model Architecture
parser.add_argument('--rnn_type', type=str, default='lstm', help='the rnn cell used')
parser.add_argument('--hidden_size', type=int, default=256, help='the hidden size of RNNs [256]')
parser.add_argument('--embedding_size', type=int, default=200, help='the embedding size [200]')
parser.add_argument('--vocab_size', type=int, default=25000, help='the vocab size [25000]')
# Training hyperparameter
parser.add_argument('--word_base', action='store_true')
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of encoder [1e-2]')
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--valid_iters', type=int, default=1, help='run validation batch every N epochs ')
parser.add_argument('--max_sent', type=int, default=25, help='max length of encoder, decoder')
parser.add_argument('--display_iters', type=int, default=10, help='display training status every N iters [25]')
parser.add_argument('--save_freq', type=int, default=5, help='save model every N epochs')
parser.add_argument('--epoch', type=int, default=25, help='train for N epochs [20]')
parser.add_argument('--init_embedding', action='store_true', help='whether init embedding')
parser.add_argument('--embedding_source', type=str, default='./', help='pretrained embedding path')

args = parser.parse_args()

if __name__ == '__main__':
    train = data.load_data('train.json', args.word_base)
    test = data.load_data('test.json', args.word_base)
    vocabulary, pad_lens = data.build_vocab(train, test, args.vocab_size)
    print('Vocab size: %d | Max context: %d | Max question: %d'%(
          len(vocabulary), pad_lens[0], pad_lens[1]))
    train, valid = data.split_exp(train, args.valid_ratio)
    print('Train: %d | Valid: %d | Test: %d'%(len(train), len(valid), len(test)))
    valid_engine = DataLoader(data.DataEngine(valid, vocabulary, pad_lens),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=use_cuda)
    test_engine = data.DataEngine(test, vocabulary, pad_lens)
    
    fusion_net = torch.load('model.cpt')

    if use_cuda:
        fusion_net = fusion_net.cuda()
    fusion_net.eval()
    
    valid_f1, valid_exact = 0, 0
    for context, q, ans_offset in valid_engine:
        context = Variable(context).cuda() if use_cuda else Variable(context)
        q = Variable(q).cuda() if use_cuda else Variable(q)
        start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
        end_ans = Variable(ans_offset[:, 1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
        start, end, start_attn, end_attn = fusion_net(context, q)

        start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1, 20)
        f1_score, exact_match_score = batch_score(start, end, ans_offset)
        valid_f1 += f1_score
        valid_exact += exact_match_score
    print('valid_f1: %f | valid_exact: %f'%(
          valid_f1/len(valid_engine), valid_exact/len(valid_engine)
        ))

    f = open('predict.csv', 'w')
    f.write('id,answer\n')
    for i in tqdm(range(len(test_engine))):
        context, q, ans_offset = test_engine[i]
        context = Variable(context).cuda() if use_cuda else Variable(context)
        q = Variable(q).cuda() if use_cuda else Variable(q)
        start, end, start_attn, end_attn = fusion_net(context.unsqueeze(0), q.unsqueeze(0))

        #max_len = len(test_engine.datas[i]['context'])
        max_len = 20
        start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1, max_len)
        qa_id = test_engine.datas[i]['id']
        output = qa_id + ','
        for x in range(start[0][0], end[0][0]):
            output += str(x) + ' '
        output += '\n'
        f.write(output)
    
