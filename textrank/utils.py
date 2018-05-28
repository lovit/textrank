from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
import numpy as np

def sents_to_wordgraph(sents, tokenizer=lambda s:s.split(),
    min_count=10, min_cooccurrence=3, verbose=True):

    vocab_counter, idx2vocab = _scan_vocabulary(sents, tokenizer, min_count)
    cooccurrence = defaultdict(lambda: defaultdict(int))
    n_sents = len(sents)

    for i_sent, sent in enumerate(sents):

        words = tokenizer(sent)
        n = len(words)
        if n < 2:
            continue

        for left, right in zip(words, words[1:]):
            if not (left in vocab_counter) or not (right in vocab_counter):
                continue
            cooccurrence[left][right] += 1
            cooccurrence[right][left] += 1

        if verbose and i_sent % 100 == 0:
            print('\rconstruct word graph {} / {} sents'.format(
                i_sent, n_sents), end='', flush=True)

    if verbose:
        print('\rconstructing word graph from {} sents was done'.format(i_sent+1), flush=True)

    cooccurrence = {left:{right:count for right, count in rdict.items()
                          if count >= min_cooccurrence}
                    for left, rdict in cooccurrence.items()}


    cooccurrence = {left:rdict for left, rdict in cooccurrence.items() if rdict}
    vocab2idx = {vocab:idx for idx, vocab in enumerate(idx2vocab)}
    cooccurrence = _encode_cooccurrence(cooccurrence, vocab2idx)

    return cooccurrence, idx2vocab
    
def sents_to_sentgraph(sents, tokenizer=lambda s:s.split(), vocab2idx=None,
    min_count=10, min_similarity=0.4, min_length=4, verbose=True):

    if not vocab2idx:
        _, idx2vocab = _scan_vocabulary(sents, tokenizer, min_count)
        vocab2idx = {vocab:idx for idx, vocab in enumerate(idx2vocab)}

    sents_ = [{vocab2idx[vocab] for vocab in tokenizer(sent) if vocab in vocab2idx}
              for sent in sents]

    rows = []
    cols = []
    data = []
    n_sents = len(sents_)

    for i, sent_i in enumerate(sents_):

        if verbose and i % 10 == 0:
            print('\rconstruct sent graph {} / {} sents'.format(
                i, n_sents), end='', flush=True)

        len_i = len(sent_i)
        if len_i < min_length:
            continue

        for j, sent_j in enumerate(sents_):

            if i == j:
                continue
            len_j = len(sent_j)
            if len_j < min_length:
                continue

            sim = len(sent_i.intersection(sent_j)) / (np.log(len_i) + np.log(len_j))
            if sim < min_similarity:
                continue
            rows.append(i)
            cols.append(j)
            data.append(sim)

    if verbose:
        print('\rconstructing sent graph from {} sents was done.'.format(n_sents), flush=True)

    return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))

def _scan_vocabulary(sents, tokenizer, min_count):
    vocab_counter = Counter(vocab for sent in sents for vocab in tokenizer(sent))
    vocab_counter = {vocab:count for vocab, count in vocab_counter.items()
                     if count >= min_count}
    vocab2idx = {vocab:idx for idx, vocab in enumerate(sorted(
        vocab_counter, key=lambda x:-vocab_counter[x]))}
    idx2vocab = [vocab for vocab in sorted(vocab2idx, key=lambda x:vocab2idx[x])]
    return vocab_counter, idx2vocab

def _encode_cooccurrence(cooccurrence, vocab2idx):
    rows = []
    cols = []
    data = []
    for vocab1, vocab2s in cooccurrence.items():
        vocab1 = vocab2idx[vocab1]
        for vocab2, count in vocab2s.items():
            vocab2 = vocab2idx[vocab2]
            rows.append(vocab1)
            cols.append(vocab2)
            data.append(count)
    n_vocabs = max(max(rows), max(cols)) + 1
    return csr_matrix((data, (rows, cols)), shape=(n_vocabs, n_vocabs))

def bow_to_graph(x):
    """It transform doc-term sparse matrix to graph.
    Vertex = [doc_0, doc_1, ..., doc_{n-1}|term_0, term_1, ..., term_{m-1}]

    Arguments
    ---------
    x: scipy.sparse

    Returns
    -------
    g: scipy.sparse.csr_matrix
        V` = x.shape[0] + x.shape[1]
        its shape = (V`, V`)
    """
    x = x.tocsr()
    x_ = x.transpose().tocsr()
    data = np.concatenate((x.data, x_.data))
    indices = np.concatenate(
        (x.indices + x.shape[0] , x_.indices))
    indptr = np.concatenate(
        (x.indptr, x_.indptr[1:] + len(x.data)))
    return csr_matrix((data, indices, indptr))

def matrix_to_dict(m):
    """It transform sparse matrix (scipy.sparse.matrix) to dictdict"""
    d = defaultdict(lambda: {})
    for f, (idx_b, idx_e) in enumerate(zip(m.indptr, m.indptr[1:])):
        for idx in range(idx_b, idx_e):
            d[f][m.indices[idx]] = m.data[idx]
    return dict(d)

def dict_to_matrix(dd):
    rows = []
    cols = []
    data = []
    for d1, d2s in dd.items():
        for d2, w in d2s.items():
            rows.append(d1)
            cols.append(d2)
            data.append(w)
    n_nodes = max(max(rows), max(cols)) + 1
    x = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return x

def is_dict_dict(dd):
    if not type(dd) == dict:
        return False
    value_item = list(dd.values())[0]
    return type(value_item) == dict

def is_numeric_dict_dict(dd):
    if not is_dict_dict(dd):
        return False
    key0 = list(dd.keys())[0]
    key1, value1 = list(list(dd.values())[0].items())[0]
    if not type(key0) == int or not type(key1) == int:
        return False
    return type(value1) == int or type(value1) == float