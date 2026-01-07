import numpy as np

from level_3.utils_level_3.psycho_utils import get_spreading_tables, process_frame_fft


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
                    Dimensions: 42x8 for EIGHT_SHORT_SEQUENCE frames, 69x1 for all other types
    """
    
    # Get spreading function tables
    tables = get_spreading_tables()
    spreading_long = tables['spreading_long']
    spreading_short = tables['spreading_short']
    bval_long = tables['bval_long']
    bval_short = tables['bval_short']
    
    # Determine if we're using short or long windows
    if frame_type == 'ESH':  # EIGHT_SHORT_SEQUENCE
        num_bands = len(bval_short)
        num_windows = 8
        SMR = np.zeros((num_bands, num_windows))
        
        # Process each of the 8 sub-frames (256 samples each) for current and 2 previous frames
        # For EIGHT_SHORT: frame_T has 8 sub-frames of 256 samples
        # We need to maintain 3 frames worth of sub-frames (current + 2 previous)
        
        # Process all 3 frames' sub-frames and store their FFT analysis
        # Each frame has 8 sub-frames, each sub-frame is 256 samples
        all_subframes = []
        
        for frame in [frame_T_prev_2, frame_T_prev_1, frame_T]:
            for i in range(num_windows):
                start_idx = i * 128
                end_idx = start_idx + 256
                subframe = frame[start_idx:end_idx]
                r, f = process_frame_fft(subframe)
                all_subframes.append({'r': r, 'f': f})
        
        # TODO: Continue with step 3 of psychoacoustic model for short frames
        
    else:  # OLS, LSS, LPS (long frames)
        num_bands = len(bval_long)
        SMR = np.zeros((num_bands, 1))
        
        # Process the 3 frames (current + 2 previous) - each is 2048 samples
        # Apply Hann window and compute FFT for each frame
        r_prev_2, f_prev_2 = process_frame_fft(frame_T_prev_2)
        r_prev_1, f_prev_1 = process_frame_fft(frame_T_prev_1)
        r_current, f_current = process_frame_fft(frame_T)
        
        # TODO: Continue with step 3 of psychoacoustic model for long frames
    
    return SMR
