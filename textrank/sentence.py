from collections import Counter
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

from .utils import scan_vocabulary
from .utils import tokenize_sents


def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3,
    similarity=None, vocab_to_idx=None, verbose=False):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(sent) return list of str
    min_count : int
        Minimum term frequency
    min_sim : float
        Minimum similarity between sentences
    similarity : callable or str
        similarity(s1, s2) returns float
        s1 and s2 are list of str.
        available similarity = [callable, 'cosine', 'textrank']
    vocab_to_idx : dict
        Vocabulary to index mapper.
        If None, this function scan vocabulary first.
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    sentence similarity graph : scipy.sparse.csr_matrix
        shape = (n sents, n sents)
    """

    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x:x[1])]

    x = vectorize_sents(sents, tokenize, vocab_to_idx)
    if similarity == 'cosine':
        x = numpy_cosine_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    else:
        x = numpy_textrank_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    return x

def vectorize_sents(sents, tokenize, vocab_to_idx):
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenize(sent))
        for token, count in counter.items():
            j = vocab_to_idx.get(token, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

def numpy_cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
    n_rows = x.shape[0]
    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx+1) * batch_size))
        psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
        rows, cols = np.where(psim >= min_sim)
        data = psim[rows, cols]
        mat.append(csr_matrix((data, (rows, cols)), shape=(e-b, n_rows)))
        if verbose:
            print('\rcalculating cosine sentence similarity {} / {}'.format(b, n_rows), end='')
    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating cosine sentence similarity was done with {} sents'.format(n_rows))
    return mat

def numpy_textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx+1) * batch_size))

        # dot product
        inner = z[b:e,:] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1,1) + size.reshape(1,-1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e-b, n_rows)))

        if verbose:
            print('\rcalculating textrank sentence similarity {} / {}'.format(b, n_rows), end='')

    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating textrank sentence similarity was done with {} sents'.format(n_rows))

    return mat

def graph_with_python_sim(tokens, verbose, similarity, min_sim):
    if similarity == 'cosine':
        similarity = cosine_sent_sim
    elif callable(similarity):
        similarity = similarity
    else:
        similarity = textrank_sent_sim

    rows, cols, data = [], [], []
    n_sents = len(tokens)
    for i, tokens_i in enumerate(tokens):
        if verbose and i % 1000 == 0:
            print('\rconstructing sentence graph {} / {} ...'.format(i, n_sents), end='')
        for j, tokens_j in enumerate(tokens):
            if i >= j:
                continue
            sim = similarity(tokens_i, tokens_j)
            if sim < min_sim:
                continue
            rows.append(i)
            cols.append(j)
            data.append(sim)
    if verbose:
        print('\rconstructing sentence graph was constructed from {} sents'.format(n_sents))
    return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))

def textrank_sent_sim(s1, s2):
    """
    Arguments
    ---------
    s1, s2 : list of str
        Tokenized sentences

    Returns
    -------
    Sentence similarity : float
        Non-negative number
    """
    n1 = len(s1)
    n2 = len(s2)
    if (n1 <= 1) or (n2 <= 1):
        return 0
    common = len(set(s1).intersection(set(s2)))
    base = math.log(n1) + math.log(n2)
    return common / base

def cosine_sent_sim(s1, s2):
    """
    Arguments
    ---------
    s1, s2 : list of str
        Tokenized sentences

    Returns
    -------
    Sentence similarity : float
        Non-negative number
    """
    if (not s1) or (not s2):
        return 0

    s1 = Counter(s1)
    s2 = Counter(s2)
    norm1 = math.sqrt(sum(v ** 2 for v in s1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in s2.values()))
    prod = 0
    for k, v in s1.items():
        prod += v * s2.get(k, 0)
    return prod / (norm1 * norm2)
