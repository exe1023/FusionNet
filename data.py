import json
import torch
from torch.utils.data import Dataset
from collections import Counter


PAD = 0
UNK = 1

def load_data(file_name, word_base):
    '''
    load train, test file
    Add other preprocessing?
    '''
    examples = json.load(open(file_name, 'r'))
    for idx in range(len(examples)):
        if word_base:
            context = examples[idx]['context_token']
            q = examples[idx]['q_token']
        else:
            context = list(examples[idx]['context'])
            q = list(examples[idx]['q'])
        examples[idx]['context'] = context
        examples[idx]['q'] = q
        
        appear = []
        for w in context:
            if w in q:
                appear.append(1)
            else:
                appear.append(0)
        examples[idx]['appear'] = appear
    return examples

def split_exp(examples, ratio):
    split_num = int(len(examples) * ratio)
    return examples[:-split_num], examples[-split_num:]

def count_vocab(examples):
    vocab_count = Counter()
    max_context, max_q = 0, 0 
    for example in examples:
        context = example['context']
        q = example['q']
        vocab_count.update(context)
        vocab_count.update(q)

        max_context = max(max_context, len(context))
        max_q = max(max_q, len(q))

    return vocab_count, (max_context, max_q)

class Vocabulary:
    def __init__(self, to_word, to_idx):
        self.to_word = to_word
        self.to_idx = to_idx
    
    def __len__(self):
        return len(self.to_word)
    
    def idx2word(self, idxs):
        words = []
        for idx in idxs:
            words.append(self.to_word[idx])
        return words
    
    def word2idx(self, words):
        idxs = []
        for word in words:
            idxs.append(self.to_idx[word])
        return idxs

def build_vocab(train, test, vocab_size):
    vocab_count, pad_lens = count_vocab(train + test)
    vocab = vocab_count.most_common()[:vocab_size]
    
    to_word, to_idx = {}, {}
    to_word[0], to_idx['<PAD>'] = '<PAD>', 0
    to_word[1], to_idx['<UNK>'] = '<UNK>', 1
    for w, c in vocab:
        to_word[len(to_word)] = w
        to_idx[w] = len(to_idx)
    return Vocabulary(to_word, to_idx), pad_lens

class DataEngine(Dataset):
    def __init__(self, datas, vocabulary, pad_lens):
        self.datas = datas
        self.vocabulary = vocabulary
        self.pad_context, self.pad_q = pad_lens
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.vectorize(self.datas[idx]['context'],
                              self.datas[idx]['q'],
                              self.datas[idx]['ans_start'],
                              self.datas[idx]['ans_end'],
                              self.datas[idx]['appear'])

    def vectorize(self, context, q, ans_start, ans_end, appear):
        padding_context = ['<PAD>' for _ in range(self.pad_context - len(context))]
        context = context + padding_context

        padding_q = ['<PAD>' for _ in range(self.pad_q - len(q))]
        q = q + padding_q

        padding_appear = [0 for _ in range(self.pad_context - len(appear))]
        appear = appear + padding_appear

        context = torch.LongTensor(self.vocabulary.word2idx(context))
        q = torch.LongTensor(self.vocabulary.word2idx(q))
        ans_offset = torch.LongTensor([ans_start, ans_end])
        appear = torch.FloatTensor(appear)
        return context, q, ans_offset, appear



if __name__ == '__main__':
    train = load_data('train.json')
    test = load_data('test.json')
    print(len(train), len(test))
