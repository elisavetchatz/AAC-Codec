import numpy as np
from functools import lru_cache

@lru_cache(maxsize=4)
def _get_mdct_matrix(N):
    """Precompute and cache MDCT transform matrix for given N.
    
    Returns matrix C of shape (N//2, N) where:
        C[k, n] = 2 * cos(2*pi/N * (n + n0) * (k + 0.5))
    with n0 = (N/2 + 1)/2
    """
    n0 = (N / 2 + 1) / 2.0
    k = np.arange(N // 2)[:, np.newaxis]  # (N//2, 1)
    n = np.arange(N)[np.newaxis, :]        # (1, N)
    C = 2.0 * np.cos(2.0 * np.pi / N * (n + n0) * (k + 0.5))
    return C

def mdct(x):
    """
    Modified Discrete Cosine Transform (MDCT).
    
    Args:
        x (array): Input signal of length N
    Returns:
        X (array): MDCT coefficients of length N/2
    """
    x = np.asarray(x).flatten()  # Ensure 1-D input
    N = len(x)
    C = _get_mdct_matrix(N)
    X = C @ x  # Matrix-vector multiplication: (N//2, N) @ (N,) = (N//2,)
    return X.flatten()  # Ensure 1-D output
