import numpy as np
from tqdm import tqdm
import pickle

def load_embedding(data_path, to_idx, embedding_size):
    '''
    Args:
        data_path: path to embedding file
        to_idx: dict, word --> index
    '''
    word_count = 0
    with open(data_path, 'r') as f:
        word_num, dim = f.readline().strip().split()
        word_num, dim = int(word_num), int(dim)
        init_w = np.random.uniform(-0.25,0.25,(len(to_idx), dim))

        for line in tqdm(f, desc='load embedding'):
            split_line = line.strip().split()
            word, vec = split_line[0], split_line[1:]
            if len(vec) != dim:
                # error due to unicode and we don't need it 
                continue
            if word in to_idx.keys():
                init_w[to_idx[word]] = np.array(list(map(float, vec)))
                word_count += 1
    print('Found pretrained embedding:', word_count)
    
    # PCA decomposition to reduce word2vec dimensionality
    # copy from other seq2seq code which works well but I never review it
    if embedding_size < dim:
        print('Reduce %d embedding dim to %d'%(dim, embedding_size))
        U, s, Vt = np.linalg.svd(init_w, full_matrices=False)
        S = np.zeros((dim, dim))
        S[:dim, :dim] = np.diag(s)
        init_w = np.dot(U[:, :embedding_size], S[:embedding_size, :embedding_size])

    return init_w