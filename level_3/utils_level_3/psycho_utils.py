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
