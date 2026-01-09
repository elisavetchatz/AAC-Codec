import numpy as np

def create_sin_window(N):
    """
    Create sinusoid window.
    
    Args:
        N (int): Window length
        
    Returns:
        w (array): SIN window of length N
    """
    w = np.zeros(N)
    
    # Left half: WSIN_LEFT
    for n in range(N // 2):
        w[n] = np.sin(np.pi / N * (n + 0.5))
    
    # Right half: WSIN_RIGHT
    for n in range(N // 2, N):
        w[n] = np.sin(np.pi / N * (n + 0.5))
    
    return w
