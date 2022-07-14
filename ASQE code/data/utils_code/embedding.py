import json 
import numpy as np

def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

if __name__ == "__main__":
    embed_dim = 300
    word2idx = json.load(open('word_idx.json'))

    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
    word_vec = _load_word_vec('./glove.840B.300d.txt', word2idx=word2idx, embed_dim=embed_dim)
    # print(word_vec)
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = vec

    np.save('emb.vec.npy', embedding_matrix)
    