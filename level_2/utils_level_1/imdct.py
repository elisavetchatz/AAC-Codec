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
    n = np.arange(N)[:, np.newaxis]        # (N, 1)
    k = np.arange(N // 2)[np.newaxis, :]   # (1, N//2)
    C = (2.0 / N) * np.cos(2.0 * np.pi / N * (n + n0) * (k + 0.5))
    return C

def imdct(X):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT).
    
    Args:
        X (array): MDCT coefficients of length N/2
        
    Returns:
        x (array): Reconstructed signal of length N
    """
    X = np.asarray(X).flatten()  # Ensure 1-D input
    N = len(X) * 2  # Original signal length
    C = _get_imdct_matrix(N)
    x = C @ X  # Matrix-vector multiplication: (N, N//2) @ (N//2,) = (N,)
    return x.flatten()  # Ensure 1-D output
