import numpy as np
from scipy.io import loadmat


# Cache for pre-calculated spreading function tables
_spreading_tables_cache = None

def load_bark_tables():
    """
    Load the Bark scale tables from TableB219.mat file.
    
    Returns:
        tuple: (B219a, B219b) arrays containing:
            - B219a: Table for long frames (2048 samples)
            - B219b: Table for short frames (256 samples)
    """
    mat_data = loadmat("TableB219.mat")

    b219a_array = mat_data['B219a']
    b219b_array = mat_data['B219b']

    return b219a_array, b219b_array


def spreading_function(i, j, bval):
    """
    Calculate the spreading function value between two bands.
    
    Args:
        i (int): Index of the masking band (source)
        j (int): Index of the masked band (target)
        bval (array): Array of central frequencies (bark values) for each band
        
    Returns:
        float: Spreading function value
    """
    # tmpx
    if i >= j:
        tmpx = 3.0 * (bval[j] - bval[i])
    else:
        tmpx = 1.5 * (bval[j] - bval[i])
    
    # tmpz
    tmpz = 8 * min((tmpx - 0.5)**2 - 2*(tmpx - 0.5), 0)
    
    # tmpy
    tmpy = (15.811389 + 7.5*(tmpx + 0.474) - 
            17.5*np.sqrt(1.0 + (tmpx + 0.474)**2))
    
    if tmpy < -100:
        x = 0
    else:
        x = 10**((tmpz + tmpy) / 10)
    
    return x


def calculate_spreading_function_table(bval):
    """
    Pre-calculate the spreading function for all band combinations.
    
    Args:
        bval (array): Array of central frequencies (bark values) for each band
        
    Returns:
        np.ndarray: 2D array where element [i, j] contains the spreading
                    function value from band i to band j
    """
    num_bands = len(bval)
    spread_table = np.zeros((num_bands, num_bands))
    
    for i in range(num_bands):
        for j in range(num_bands):
            spread_table[i, j] = spreading_function(i, j, bval)
    
    return spread_table


def get_spreading_tables():
    """
    Get pre-calculated spreading function tables for long and short frames.

    Returns:
        dict:
            - 'spreading_long': Spreading function table for long frames (69x69)
            - 'spreading_short': Spreading function table for short frames (42x42)
            - 'bval_long': Central frequencies for long frames
            - 'bval_short': Central frequencies for short frames
    """
    global _spreading_tables_cache
    
    if _spreading_tables_cache is None:

        B219a, B219b = load_bark_tables()
        
        # Extract bval 
        bval_long = B219a[:, 2] 
        bval_short = B219b[:, 2]
        
        spreading_long = calculate_spreading_function_table(bval_long)
        spreading_short = calculate_spreading_function_table(bval_short)
        
        # Cache the results
        _spreading_tables_cache = {
            'spreading_long': spreading_long,
            'spreading_short': spreading_short,
            'bval_long': bval_long,
            'bval_short': bval_short
        }
    
    return _spreading_tables_cache


def apply_hann_window(signal):
   
    N = len(signal)
    n = np.arange(N)

    hann_window = 0.5 - 0.5 * np.cos(np.pi * (n + 0.5) / N)

    return signal * hann_window


def compute_fft_analysis(signal):
    """
    Compute FFT and extract magnitude and phase.
    
    For long frames (2048 samples): returns coefficients 0-1023
    For short frames (256 samples): returns coefficients 0-127
    
    Args:
        signal (array): Input signal already windowed
        
    Returns:
        (r, f) where:
            - r: magnitude for each frequency bin
            - f: phase for each frequency bin (in radians)
    """

    # FFT
    fft_result = np.fft.fft(signal)
    
    # For 2048: keep 0-1023, for 256: keep 0-127
    num_coeffs = len(signal) // 2
    fft_result = fft_result[:num_coeffs]
    
    r = np.abs(fft_result)
    f = np.angle(fft_result)
    
    return r, f


def process_frame_fft(frame):
    """
    Process a single frame/subframe: apply Hann window and compute FFT
    """

    windowed = apply_hann_window(frame)
    r, f = compute_fft_analysis(windowed)
    
    return r, f


def compute_predictions(r_prev_2, f_prev_2, r_prev_1, f_prev_1):
    """
    Compute predictions for magnitude and phase using linear extrapolation
        
    Returns:
        tuple: (rpred, fpred) predicted magnitude and phase arrays
    """
    rpred = 2 * r_prev_1 - r_prev_2
    fpred = 2 * f_prev_1 - f_prev_2
    
    return rpred, fpred


def compute_predictability(r, f, rpred, fpred):
    """
    Returns:
        array: Predictability measure c for each frequency bin
    """
    # polar to Cartesian
    real_current = r * np.cos(f)
    imag_current = r * np.sin(f)
    
    real_pred = rpred * np.cos(fpred)
    imag_pred = rpred * np.sin(fpred)
    
    numerator = (real_current - real_pred)**2 + (imag_current - imag_pred)**2
    
    # Calculate the denominator with small epsilon to avoid division by zero
    denominator = r + np.abs(rpred)
    epsilon = 1e-10
    denominator = np.maximum(denominator, epsilon)
    
    # predictability measure
    c = numerator / denominator
    
    return c
