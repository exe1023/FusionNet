import argparse
import data
from module import FusionNet, decode
from metrics import batch_score
from embedding import load_embedding
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
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--rnn_layer', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of encoder [1e-3]')
parser.add_argument('--valid_ratio', type=float, default=0.05)
parser.add_argument('--valid_iters', type=int, default=1, help='run validation batch every N epochs [1]')
parser.add_argument('--max_sent', type=int, default=25, help='max length of encoder, decoder')
parser.add_argument('--display_freq', type=int, default=10, help='display training status every N iters [10]')
parser.add_argument('--save_freq', type=int, default=1, help='save model every N epochs [1]')
parser.add_argument('--epoch', type=int, default=25, help='train for N epochs [25]')
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
    train_engine = DataLoader(data.DataEngine(train, vocabulary, pad_lens),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=use_cuda)
    valid_engine = DataLoader(data.DataEngine(valid, vocabulary, pad_lens),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=use_cuda)
    test_engine = data.DataEngine(test, vocabulary, pad_lens)
    
    if args.init_embedding:
        w2v = load_embedding(args.embedding_source, 
                       vocabulary.to_idx,
                       300)
    else:
        w2v = None
    fusion_net = FusionNet(vocab_size=len(vocabulary),
                           word_dim=300,
                           hidden_size=125,
                           rnn_layer=args.rnn_layer,
                           dropout=args.dropout,
                           pretrained_embedding=w2v)
    if use_cuda:
        fusion_net = fusion_net.cuda()
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adamax(fusion_net.parameters())
    
    for epoch in range(args.epoch):
        batch = 0
        fusion_net.train()
        for context, q, ans_offset, appear in train_engine:
            context = Variable(context).cuda() if use_cuda else Variable(context)
            q = Variable(q).cuda() if use_cuda else Variable(q)
            start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
            end_ans = Variable(ans_offset[:, 1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
            appear = Variable(appear).cuda() if use_cuda else Variable(appear)

            start, end, start_attn, end_attn = fusion_net(context, q, appear)
            loss = criterion(start_attn, start_ans) + criterion(end_attn, end_ans)
            loss.backward()
            nn.utils.clip_grad_norm(fusion_net.parameters(), 10)
            optimizer.step()

            start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1)
            f1_score, exact_match_score = batch_score(start, end, ans_offset)
            if batch % args.display_freq == 0:
                print('epoch: %d | batch: %d/%d| loss: %f | f1: %f | exact: %f'%(
                    epoch, batch, len(train_engine), loss.data[0], 
                    f1_score, exact_match_score
                ))
            batch +=1
        
        valid_f1, valid_exact = 0, 0
        fusion_net.eval()
        for context, q, ans_offset, appear in valid_engine:
            context = Variable(context).cuda() if use_cuda else Variable(context)
            q = Variable(q).cuda() if use_cuda else Variable(q)
            start_ans = Variable(ans_offset[:, 0]).cuda() if use_cuda else Variable(ans_offset[:, 0])
            end_ans = Variable(ans_offset[:, 1]).cuda() if use_cuda else Variable(ans_offset[:, 1])
            appear = Variable(appear).cuda() if use_cuda else Variable(appear)

            start, end, start_attn, end_attn = fusion_net(context, q, appear)
            start, end, scores = decode(start.data.cpu(), end.data.cpu(), 1)
            f1_score, exact_match_score = batch_score(start, end, ans_offset)
            valid_f1 += f1_score
            valid_exact += exact_match_score
        print('epoch: %d | valid_f1: %f | valid_exact: %f'%(
                  epoch, valid_f1/len(valid_engine), valid_exact/len(valid_engine)
            ))
        if epoch % args.save_freq == 0:
            torch.save(fusion_net, 'model.cpt')
    torch.save(fusion_net, 'model.final')
