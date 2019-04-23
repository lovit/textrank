from collections import Counter
import math
from scipy.sparse import csr_matrix

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
    similarity : callable
        similarity(s1, s2) returns float
        s1 and s2 are list of str.
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

    if similarity is None:
        similarity = textrank_sent_sim

    tokens = tokenize_sents(sents, tokenize)
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
    print('\rrconstructing sentence graph was constructed from {} sents'.format(n_sents))
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
