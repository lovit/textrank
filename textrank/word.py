from collections import defaultdict
from scipy.sparse import csr_matrix

from .utils import scan_vocabulary
from .utils import tokenize_sents


def word_graph(sents, tokenize=None, min_count=2, window=2,
    min_cooccurrence=2, vocab_to_idx=None, verbose=False):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(str) returns list of str
    min_count : int
        Minumum term frequency
    window : int
        Co-occurrence window size
    min_cooccurrence : int
        Minimum cooccurrence frequency
    vocab_to_idx : dict
        Vocabulary to index mapper.
        If None, this function scan vocabulary first.
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    co-occurrence word graph : scipy.sparse.csr_matrix
    idx_to_vocab : list of str
        Word list corresponding row and column
    """
    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x:x[1])]

    tokens = tokenize_sents(sents, tokenize)
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence, verbose)
    return g, idx_to_vocab

def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2, verbose=False):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenized sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper
    window : int
        Co-occurrence window size
    min_cooccurrence : int
        Minimum cooccurrence frequency
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    co-occurrence matrix : scipy.sparse.csr_matrix
        shape = (n_vocabs, n_vocabs)
    """
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        if verbose and s % 1000 == 0:
            print('\rword cooccurrence counting {}'.format(s), end='')
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    if verbose:
        print('\rword cooccurrence counting from {} sents was done'.format(s+1))
    return dict_to_mat(counter, n_vocabs, n_vocabs)

def dict_to_mat(d, n_rows, n_cols):
    """
    Arguments
    ---------
    d : dict
        key : (i,j) tuple
        value : float value

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
