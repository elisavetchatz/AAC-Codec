import numpy as np

from utils_level_3.psycho_utils import get_spreading_tables, process_frame_fft, compute_predictions


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
        
        # Step 2: Process each of the 8 sub-frames (256 samples each) for current and 2 previous frames
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
        
        # Step 3: Compute predictions for each of the 8 current sub-frames
        # For subframe i of current frame, we need:
        #   - r_{-1}: subframe i-1 (wraps to previous frame if i=0)
        #   - r_{-2}: subframe i-2 (wraps to previous frames)
        # all_subframes indices:
        #   - 0-7: frame_T_prev_2 subframes
        #   - 8-15: frame_T_prev_1 subframes
        #   - 16-23: frame_T (current) subframes
        
        predictions = []
        for i in range(num_windows):
            current_idx = 16 + i  # Current subframe index in all_subframes
            prev_1_idx = current_idx - 1  # Previous subframe (automatically wraps correctly)
            prev_2_idx = current_idx - 2  # Previous-previous subframe
            
            r_prev_2 = all_subframes[prev_2_idx]['r']
            f_prev_2 = all_subframes[prev_2_idx]['f']
            r_prev_1 = all_subframes[prev_1_idx]['r']
            f_prev_1 = all_subframes[prev_1_idx]['f']
            
            rpred, fpred = compute_predictions(r_prev_2, f_prev_2, r_prev_1, f_prev_1)
            predictions.append({'rpred': rpred, 'fpred': fpred})
        
        # TODO: Continue with step 4 of psychoacoustic model for short frames
        
    else:  # OLS, LSS, LPS (long frames)
        num_bands = len(bval_long)
        SMR = np.zeros((num_bands, 1))
        
        # Step 2: Process the 3 frames (current + 2 previous) - each is 2048 samples
        # Apply Hann window and compute FFT for each frame
        r_prev_2, f_prev_2 = process_frame_fft(frame_T_prev_2)
        r_prev_1, f_prev_1 = process_frame_fft(frame_T_prev_1)
        r_current, f_current = process_frame_fft(frame_T)
        
        # Step 3: Compute predictions using the 2 previous frames
        # rpred(w) = 2*r_{-1}(w) - r_{-2}(w)
        # fpred(w) = 2*f_{-1}(w) - f_{-2}(w)
        rpred, fpred = compute_predictions(r_prev_2, f_prev_2, r_prev_1, f_prev_1)
        
        # TODO: Continue with step 4 of psychoacoustic model for long frames
    
    return SMR
