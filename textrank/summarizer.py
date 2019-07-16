import numpy as np
from .rank import pagerank
from .sentence import sent_graph
from .word import word_graph


class KeywordSummarizer:
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        Tokenize function: tokenize(str) = list of str
    min_count : int
        Minumum frequency of words will be used to construct sentence graph
    window : int
        Word cooccurrence window size. Default is -1.
        '-1' means there is cooccurrence between two words if the words occur in a sentence
    min_cooccurrence : int
        Minimum cooccurrence frequency of two words
    vocab_to_idx : dict or None
        Vocabulary to index mapper
    df : float
        PageRank damping factor
    max_iter : int
        Number of PageRank iterations
    verbose : Boolean
        If True, it shows training progress
    """
    def __init__(self, sents=None, tokenize=None, min_count=2,
        window=-1, min_cooccurrence=2, vocab_to_idx=None,
        df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        bias : None or numpy.ndarray
            PageRank bias term

        Returns
        -------
        None
        """

        g, self.idx_to_vocab = word_graph(sents,
            self.tokenize, self.min_count,self.window,
            self.min_cooccurrence, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n words = {}'.format(self.R.shape[0]))

    def keywords(self, topk=30):
        """
        Arguments
        ---------
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')
        idxs = self.R.argsort()[-topk:]
        keywords = [(self.idx_to_vocab[idx], self.R[idx]) for idx in reversed(idxs)]
        return keywords

    def summarize(self, sents, topk=30):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """
        self.train_textrank(sents)
        return self.keywords(topk)


class KeysentenceSummarizer:
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        Tokenize function: tokenize(str) = list of str
    min_count : int
        Minumum frequency of words will be used to construct sentence graph
    min_sim : float
        Minimum similarity between sentences in sentence graph
    similarity : str
        available similarity = ['cosine', 'textrank']
    vocab_to_idx : dict or None
        Vocabulary to index mapper
    df : float
        PageRank damping factor
    max_iter : int
        Number of PageRank iterations
    verbose : Boolean
        If True, it shows training progress
    """
    def __init__(self, sents=None, tokenize=None, min_count=2,
        min_sim=0.3, similarity=None, vocab_to_idx=None,
        df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.similarity = similarity
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        bias : None or numpy.ndarray
            PageRank bias term
            Shape must be (n_sents,)

        Returns
        -------
        None
        """
        g = sent_graph(sents, self.tokenize, self.min_count,
            self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

    def summarize(self, sents, topk=30, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        topk : int
            Number of key-sentences to be selected.
        bias : None or numpy.ndarray
            PageRank bias term
            Shape must be (n_sents,)

        Returns
        -------
        keysents : list of tuple
            Each tuple stands for (sentence index, rank, sentence)

        Usage
        -----
            >>> from textrank import KeysentenceSummarizer

            >>> summarizer = KeysentenceSummarizer(tokenize = tokenizer, min_sim = 0.5)
            >>> keysents = summarizer.summarize(texts, topk=30)
        """
        n_sents = len(sents)
        if isinstance(bias, np.ndarray):
            if bias.shape != (n_sents,):
                raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
        elif bias is not None:
            raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))

        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
        return keysents
