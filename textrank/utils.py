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

def scan_vocabulary(sents, min_count=2, tokenize=None):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    min_count : int
        Minumum term frequency
    tokenize : callable
        tokenize(str) returns list of str

    Returns
    -------
    idx_to_vocab : list of str
        Vocabulary list
    vocab_to_idx : dict
        Vocabulary to index mapper.
    """
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def vectorize(tokens, vocab_to_idx):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenzed sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper

    Returns
    -------
    sentence bow : scipy.sparse.csr_matrix
        shape = (n_sents, n_terms)
    """
    rows, cols, data = [], [], []
    for i, tokens_i in enumerate(tokens):
        for t, c in counter(tokens_i).items():
            j = vocab_to_idx.get(t, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(c)
    n_sents = len(tokens)
    n_terms = len(vocab_to_idx)
    x = csr_matrix((data, (rows, cols)), shape=(n_sents, n_terms))
    return x
