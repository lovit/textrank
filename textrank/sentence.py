import math

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
