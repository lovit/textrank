from .utils import sents_to_wordgraph
from .utils import sents_to_sentgraph
from .rank import BiasedReinforceRank

def summarize_as_keywords(sents, topk=50, tokenizer=lambda s:s.split(),
    min_count=10, min_cooccurrence=3, verbose=True, debug=False):

    assert topk > 0

    g, idx2vocab = sents_to_wordgraph(sents,
        tokenizer, min_count, min_cooccurrence, verbose)

    trainer = BiasedReinforceRank(verbose=verbose)
    ranks = trainer.rank(g)

    keyword_idxs = ranks.argsort()[::-1][:topk]
    keyword_rank = ranks[keyword_idxs]
    keywords = [(idx2vocab[idx], rank) for idx, rank
                in zip(keyword_idxs, keyword_rank)]

    if not debug:
        return keywords
    return keywords, ranks, idx2vocab, g

def summarize_as_keysentences(sents, topk=5, tokenizer=lambda s:s.split(),
    vocab2idx=None, min_count=10, min_similarity=0.4, min_length=4, verbose=True):

    assert topk > 0

    g = sents_to_sentgraph(sents, tokenizer, vocab2idx,
            min_count, min_similarity, min_length, verbose)

    trainer = BiasedReinforceRank(verbose=verbose)
    ranks = trainer.rank(g)
    
    keysent_idxs = ranks.argsort()[::-1][:topk]
    keysent_rank = ranks[keysent_idxs]

    keysents = [(sents[idx], rank) for idx, rank
                in zip(keysent_idxs, keysent_rank)]

    return keysents