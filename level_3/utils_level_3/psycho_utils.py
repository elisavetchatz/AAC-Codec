import numpy as np
import os
import sys

# Add parent directories to path to import from level_2
current_dir = os.path.dirname(os.path.abspath(__file__))
level_3_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(level_3_dir)
sys.path.insert(0, root_dir)

from level_2.utils_level_2.tns_utils import load_band_tables


# Cache for pre-calculated spreading function tables
_spreading_tables_cache = None

def load_bark_tables():
    """
    Uses the existing function from tns_utils
    """
    return load_band_tables()


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
            - 'wlow_long': Lower frequency indices for long frame bands
            - 'whigh_long': Upper frequency indices for long frame bands
            - 'wlow_short': Lower frequency indices for short frame bands
            - 'whigh_short': Upper frequency indices for short frame bands
    """
    global _spreading_tables_cache
    
    if _spreading_tables_cache is None:

        B219a, B219b = load_bark_tables()
        
        # Extract columns from tables
        # Column 0: band index
        # Column 1: wlow (lower frequency index)
        # Column 2: bval (central frequency)
        # Column 3: whigh (upper frequency index)
        # Column 4: width
        
        bval_long = B219a[:, 2] 
        bval_short = B219b[:, 2]
        
        wlow_long = B219a[:, 1].astype(int)
        whigh_long = B219a[:, 3].astype(int)
        wlow_short = B219b[:, 1].astype(int)
        whigh_short = B219b[:, 3].astype(int)
        
        spreading_long = calculate_spreading_function_table(bval_long)
        spreading_short = calculate_spreading_function_table(bval_short)
        
        # Cache the results
        _spreading_tables_cache = {
            'spreading_long': spreading_long,
            'spreading_short': spreading_short,
            'bval_long': bval_long,
            'bval_short': bval_short,
            'wlow_long': wlow_long,
            'whigh_long': whigh_long,
            'wlow_short': wlow_short,
            'whigh_short': whigh_short
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


def compute_band_energy_predictability(r, c, wlow, whigh):
    """
    Compute energy and weighted predictability for each critical band
        
    Returns:
        tuple: (e_bands, c_bands)
    """

    num_bands = len(wlow)
    e_bands = np.zeros(num_bands)
    c_bands = np.zeros(num_bands)
    
    for b in range(num_bands):

        w_start = wlow[b]
        w_end = whigh[b] + 1
        
        r_band = r[w_start:w_end]
        c_band = c[w_start:w_end]
        
        r_squared = r_band ** 2
        
        e_bands[b] = np.sum(r_squared)
        
        # Compute weighted predictability
        c_bands[b] = np.sum(c_band * r_squared)
    
    return e_bands, c_bands


def apply_spreading_function(e_bands, c_bands, spreading_table):
    """    
    Returns:
        tuple: (cb, en)
            - cb: Normalized predictability for each band
            - en: Normalized energy for each band
    """
    ecb = e_bands @ spreading_table
    
    ct = c_bands @ spreading_table 
    
    # Normalize predictability: cb(b) = ct(b) / ecb(b)
    epsilon = 1e-10
    cb = ct / np.maximum(ecb, epsilon)
    
    # Normalize energy
    spreading_sum = np.sum(spreading_table, axis=0)
    en = ecb / np.maximum(spreading_sum, epsilon)
    
    return cb, en
