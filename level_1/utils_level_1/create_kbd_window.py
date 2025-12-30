import numpy as np
from scipy.signal.windows import kaiser

def create_kbd_window(N, alpha):
    """
    Create Kaiser-Bessel-Derived (KBD) window.
    
    Args:
        N (int): Window length
        alpha (float): Kaiser window parameter (6 for long, 4 for short)
        
    Returns:
        w (array): KBD window of length N
    """
    # Compute Kaiser-Bessel window using scipy
    # The window length should be N/2 + 1 for proper construction
    kaiser_window = kaiser(N // 2 + 1, alpha * np.pi, sym=True)
    
    # # Compute cumulative sum for KBD
    # w_intermediate = np.zeros(N // 2 + 1)
    # for i in range(N // 2 + 1):
    #     w_intermediate[i] = kaiser_window[i]
    
    # Cumulative sums
    cumsum = np.cumsum(kaiser_window)
    norm_factor = cumsum[-1]
    
    # Create left and right halves
    w = np.zeros(N)
    
    # Left half: WKBD_LEFT
    for n in range(N // 2):
        w[n] = cumsum[n] / norm_factor
    
    # Right half: WKBD_RIGHT
    for n in range(N // 2, N):
        w[n] = cumsum[N - n - 1] / norm_factor
    
    return w