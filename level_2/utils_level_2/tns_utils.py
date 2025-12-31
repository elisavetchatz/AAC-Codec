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
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    mat_file = os.path.join(root_dir, 'TableB219.mat')
    
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
    Sw = np.zeros(N)
    
    for j in range(num_bands):
        bj = band_starts[j]
        
        if j < num_bands - 1:
            bj_next = band_starts[j + 1]
        else:
            bj_next = N
        
        # S_w(k) = sqrt(P(j)) for all k in band j
        Sw[bj:bj_next] = np.sqrt(P[j])
    
    # Smoothing: backward pass (1022 down to 0)
    for k in range(N - 2, -1, -1):
        Sw[k] = (Sw[k] + Sw[k + 1]) / 2.0
    
    # Smoothing: forward pass (1 up to N-1)
    for k in range(1, N):
        Sw[k] = (Sw[k] + Sw[k - 1]) / 2.0
    
    return Sw

def solve_lpc_coeffs(Xw, order=4):
    """
    Solve for Linear Prediction Coefficients using autocorrelation method.
    
    Solves the normal equations: R * a = r
    where R is the autocorrelation matrix and r is the autocorrelation vector.
    
    Args:
        X_w: Normalized MDCT coefficients
        order: LPC filter order (default 4)
    
    Returns:
        a: LPC coefficients [a1, a2, ..., a_p]
    """

    Xw = np.asarray(Xw).flatten()
    p = order
    
    # Compute autocorrelation values r(0), r(1), ..., r(p)
    r = np.zeros(p + 1)
    for lag in range(p + 1):
        r[lag] = np.sum(Xw[lag:] * Xw[:len(Xw) - lag])
    
    # Build autocorrelation matrix R (Toeplitz structure)
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    
    # Right-hand side vector
    r_vec = r[1:p + 1]
    
    # Solve normal equations
    try:
        a = np.linalg.solve(R + 1e-10 * np.eye(p), r_vec)
        
    except np.linalg.LinAlgError:
        # If singular, return zero coefficients
        a = np.zeros(p)
    
    return a

def quantize_tns_coeffs(a, step=0.1):
    """
    Quantize LPC coefficients using uniform symmetric quantizer.
    
    Args:
        a: LPC coefficients
        step: Quantization step size (default 0.1 for 4-bit quantization)
    
    Returns:
        a_quant: Quantized coefficients
    """

    a_quant = np.round(a / step) * step
    a_quant = np.clip(a_quant, -0.8, 0.7)
    
    return a_quant