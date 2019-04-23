from .rank import pagerank
from .word import word_graph


class KeywordSummarizer:
    def __init__(self, sents=None, tokenize=None, min_count=2,
        window=-1, min_cooccurrence=2, vocab_to_idx=None,
        df=0.85, max_iter=30, bias=None, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.bias = bias
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents):
        g, self.idx_to_vocab = word_graph(sents,
            self.tokenize, self.min_count,self.window,
            self.min_cooccurrence, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, self.bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n words = {}'.format(self.R.shape[0]))

    def keywords(self, topk=30):
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')
        keywords = [(w, r) for w, r in zip(self.idx_to_vocab, self.R)]
        return keywords

    def summarize(self, sents, topk=30):
        self.train_textrank(sents)
        return self.keywords(topk)