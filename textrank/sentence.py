from collections import Counter
import math

from .utils import scan_vocabulary


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
