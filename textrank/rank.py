import time
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from .utils import is_numeric_dict_dict
from .utils import dict_to_matrix

class BiasedReinforceRank():
    
    def __init__(self, damping_factor=0.85, max_iter=30,
        ranksum=1.0, verbose=True, converge_threshold=0.0001):
        
        self.df = damping_factor
        self.max_iter = max_iter
        self.ranksum = ranksum
        self.verbose = verbose
        self.converge_threshold = converge_threshold

    def rank(self, inbound_matrix, bias=None):
        if is_numeric_dict_dict(inbound_matrix):
            x = dict_to_matrix(inbound_matrix)
        elif not sp.issparse(inbound_matrix):
            raise ValueError('inboud_matrix type should be sparse matrix')
        else:
            x = inbound_matrix

        self.rank = pagerank(
            x, self.df, self.max_iter, bias,
            self.ranksum, self.verbose, self.converge_threshold)
        return self.rank

def pagerank(inbound_matrix, df=0.85, max_iter=30,
    bias=None, ranksum=1.0, verbose=True, converge_threshold=0.0001):

    converge_threshold_ = ranksum * converge_threshold
    n_nodes, initial_weight, rank, bias = _initialize_rank_parameters(
        inbound_matrix, df, bias, ranksum)

    for n_iter in range(1, max_iter + 1):
        t = time.time()
        rank_new = _update_pagerank(inbound_matrix, rank, bias, df, ranksum)
        t = time.time() - t

        diff = np.sqrt(((rank - rank_new) **2).sum())
        rank = rank_new

        if diff <= converge_threshold_:
            if verbose:
                print('Early stop. because it already converged.')
            break
        if verbose:
            print('iter {} : diff = {} ({} sec)'.format(n_iter, diff, '%.3f'%t))

    return rank

def _initialize_rank_parameters(inbound_matrix, df, bias, ranksum):
    # Check number of nodes and initial weight
    n_nodes = inbound_matrix.shape[0]
    initial_weight = ranksum / n_nodes

    # Initialize rank and bias
    rank = np.asarray([initial_weight] * n_nodes)    
    if not bias:
        bias = rank.copy()
    elif not isinstance(bias, np.ndarray):
        raise ValueError('bias must be numpy.ndarray type or None')

    return n_nodes, initial_weight, rank, bias

def _update_pagerank(inbound_matrix, rank, bias, df, ranksum=1.0):
    # call scipy.sparse safe_sparse_dot()
    rank_new = inbound_matrix.dot(rank)
    rank_new = normalize(rank_new.reshape(1, -1), norm='l2').reshape(-1) * ranksum
    rank_new = df * rank_new + (1 - df) * bias
    return rank_new