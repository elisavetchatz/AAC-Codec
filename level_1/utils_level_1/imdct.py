import numpy as np
from functools import lru_cache

@lru_cache(maxsize=4)
def _get_imdct_matrix(N):
    """Precompute and cache IMDCT transform matrix for given N (output length).
    
    Returns matrix C of shape (N, N//2) where:
        C[n, k] = (2/N) * cos(2*pi/N * (n + n0) * (k + 0.5))
    with n0 = (N/2 + 1)/2
    """
    n0 = (N / 2 + 1) / 2.0
    n = np.arange(N)[:, np.newaxis]
    k = np.arange(N // 2)[np.newaxis, :]
    C = (2.0 / N) * np.cos(2.0 * np.pi / N * (n + n0) * (k + 0.5))
    
    return C

def imdct(X):
    """
    Inverse Modified Discrete Cosine Transform
    """
    X = np.asarray(X).flatten()
    N = len(X) * 2
    C = _get_imdct_matrix(N)
    x = C @ X

    return x.flatten()
