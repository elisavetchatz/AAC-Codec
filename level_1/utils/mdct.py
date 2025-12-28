import numpy as np

def mdct(x):
    """
    Modified Discrete Cosine Transform (MDCT).
    
    Args:
        x (array): Input signal of length N
        
    Returns:
        X (array): MDCT coefficients of length N/2
    """
    N = len(x)
    n0 = (N / 2 + 1) / 2
    
    # MDCT formula
    X = np.zeros(N // 2)
    for k in range(N // 2):
        for n in range(N):
            X[k] += x[n] * np.cos(2 * np.pi / N * (n + n0) * (k + 0.5))
    
    X *= 2
    return X
