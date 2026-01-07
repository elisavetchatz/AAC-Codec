"""
Psychoacoustic Model for AAC Encoder Level 3.

This module implements the psychoacoustic model according to the AAC standard
(ISO/IEC 13818-7, pages 95-101 of w2203tfa).
"""

import numpy as np
from level_3.utils_level_3.psycho_utils import load_bark_tables, calculate_spreading_function_table


# Load Bark scale tables and pre-calculate spreading functions
B219a, B219b = load_bark_tables()

# Extract bval (central frequencies) from the tables
# Column indices: 0=index, 1=width, 2=bval (based on standard table structure)
bval_long = B219a[:, 2]   # For long frames (2048 samples)
bval_short = B219b[:, 2]  # For short frames (256 samples)

# Pre-calculate spreading function tables for efficiency
spreading_long = calculate_spreading_function_table(bval_long)
spreading_short = calculate_spreading_function_table(bval_short)


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic model implementation for one channel.

    Args:
        frame_T (array): Current frame in time domain
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
        frame_T_prev_1 (array): Previous frame of frame_T in the same channel
        frame_T_prev_2 (array): Frame before the previous frame of frame_T in the same channel
    
    Returns:
        SMR (array): Signal to Mask Ratio
                    Dimensions: 42×8 for EIGHT_SHORT_SEQUENCE frames, 69×1 for all other types
    """
    # Determine if we're using short or long windows
    if frame_type == 'ESH':  # EIGHT_SHORT_SEQUENCE
        num_bands = len(bval_short)
        num_windows = 8
        SMR = np.zeros((num_bands, num_windows))
        # TODO: Implement psychoacoustic model for short frames
    else:  # OLS, LSS, LPS (long frames)
        num_bands = len(bval_long)
        SMR = np.zeros((num_bands, 1))
        # TODO: Implement psychoacoustic model for long frames
    
    return SMR
