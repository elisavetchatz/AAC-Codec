import numpy as np

def imdct(X):

    """
    Inverse Modified Discrete Cosine Transform (IMDCT).
    
    Args:
        X (array): MDCT coefficients of length N/2
        
    Returns:
        x (array): Reconstructed signal of length N
    """
    
    N = len(X) * 2  # Original signal length
    n0 = (N / 2 + 1) / 2
    
    # IMDCT formula
    x = np.zeros(N)
    for n in range(N):
        for k in range(N // 2):
            x[n] += X[k] * np.cos(2 * np.pi / N * (n + n0) * (k + 0.5))
    
    x *= 2 / N

    return x
