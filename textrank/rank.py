import numpy as np
from sklearn.preprocessing import normalize

def pagerank(x, df=0.85, max_iter=30, bias=None):
    """
    Arguments
    ---------
    x : scipy.sparse.csr_matrix
        shape = (n vertex, n vertex)
    df : float
        Damping factor, 0 < df < 1
    max_iter : int
        Maximum number of iteration
    bias : numpy.ndarray or None
        If None, equal bias

    Returns
    -------
    R : numpy.ndarray
        PageRank vector. shape = (n vertex, 1)
    """

    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)

    # check bias
    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
    else:
        bias = bias.reshape(-1,1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R
