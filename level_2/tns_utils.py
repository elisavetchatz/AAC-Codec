import numpy as np
import scipy.io as sio
import os


def load_band_tables():
    """
    Load psychoacoustic model band tables from TableB219.mat
    
    Returns:
        B219a: Table for long frames (69 bands)
        B219b: Table for short frames (42 bands)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    mat_file = os.path.join(parent_dir, 'TableB219.mat')
    
    mat = sio.loadmat(mat_file)
    B219a = mat['B219a']  # For long frames (69 bands)
    B219b = mat['B219b']  # For short frames (42 bands)
    
    return B219a, B219b


def get_band_boundaries(frame_type):
    """
    Get band boundaries based on frame type.
    
    Args:
        frame_type: 'OLS', 'LSS', 'LPS', or 'ESH'
    
    Returns:
        band_starts: Array of starting indices for each band (w_low)
        num_bands: Number of bands
    """
    B219a, B219b = load_band_tables()
    
    if frame_type == 'ESH':

        # For short frames: 42 bands, Column 1 is w_low
        band_starts = B219b[:, 1].astype(int)
        num_bands = len(band_starts)
    else:
        # For long frames: 69 bands
        band_starts = B219a[:, 1].astype(int)
        num_bands = len(band_starts)
    
    return band_starts, num_bands


def compute_band_energy(X, frame_type):
    """
    Compute energy P(j) for each psychoacoustic band.
    
    Args:
        X: MDCT coefficients (1024,) for long frames or (128,) for short subframes
        frame_type: Frame type
    
    Returns:
        P: Array of band energies, shape (NB,)
    """
    X = np.asarray(X).flatten()
    band_starts, num_bands = get_band_boundaries(frame_type)
    
    P = np.zeros(num_bands)
    
    for j in range(num_bands):

        # Get band boundaries
        bj = band_starts[j]
        
        if j < num_bands - 1:
            bj_next = band_starts[j + 1]
        else:
            # Last band goes to the end
            bj_next = len(X)
        
        # Compute energy: sum of squared coefficients
        P[j] = np.sum(X[bj:bj_next] ** 2)
    
    return P


def compute_normalization_factors(X, frame_type):
    """
    Compute normalization factors S_w(k) for each MDCT coefficient.
    
    According to Eq. (2)-(4):
    1. Compute band energies P(j)
    2. Assign sqrt(P(j)) to all coefficients in band j
    3. Smooth the factors twice (forward and backward)
    
    Args:
        X: MDCT coefficients
        frame_type: Frame type
    
    Returns:
        S_w: Normalization factors, same shape as X
    """
    X = np.asarray(X).flatten()
    N = len(X)
    
    # Compute band energies
    P = compute_band_energy(X, frame_type)
    band_starts, num_bands = get_band_boundaries(frame_type)
    
    # Initialize S_w with sqrt of band energies
    S_w = np.zeros(N)
    
    for j in range(num_bands):
        b_j = band_starts[j]
        
        if j < num_bands - 1:
            b_j_next = band_starts[j + 1]
        else:
            b_j_next = N
        
        # S_w(k) = sqrt(P(j)) for all k in band j
        S_w[b_j:b_j_next] = np.sqrt(P[j])
    
    # Smoothing: backward pass (1022 down to 0)
    for k in range(N - 2, -1, -1):
        S_w[k] = (S_w[k] + S_w[k + 1]) / 2.0
    
    # Smoothing: forward pass (1 up to N-1)
    for k in range(1, N):
        S_w[k] = (S_w[k] + S_w[k - 1]) / 2.0
    
    return S_w
