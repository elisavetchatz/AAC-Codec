import numpy as np

from level_1.utils_level_1.create_kbd_window import create_kbd_window  
from level_1.utils_level_1.create_sin_window import create_sin_window  
from level_1.utils_level_1.mdct import mdct  

def filter_bank(frame_T, frame_type, win_type):
    """
    Filter Bank implementation.

    Args:
        frame_T (2048x2 array): frame in time domain
        frame_type (str): frame type of the current frame
        win_type (str): type of the weight window of the current frame. Can be 'KBD' or 'SIN'

    Returns:
        frame_F (1024x2 array): frame in frequency domain in an MDCT coefficient format
                                Consists of either:
                                a) the coefficients of the 2 channels for 'OLS', 'LSS', 'LPS' frame types or
                                b) 8 (128x2) subarrays for each subframe for 'ESH' frame type, placed in rows according to subframe order
    """
    # Create windows based on win_type
    if win_type == 'KBD':
        W_long = create_kbd_window(2048, alpha=6)
        W_short = create_kbd_window(256, alpha=4)
    elif win_type == 'SIN':
        W_long = create_sin_window(2048)
        W_short = create_sin_window(256)
    else:
        raise ValueError(f"Unknown window type: {win_type}")
    
    # Process based on frame_type
    if frame_type == 'ONLY_LONG_SEQUENCE' or frame_type == 'OLS':

        # Use symmetric long window for entire frame
        frame_F = np.zeros((1024, 2))
        for ch in range(2):
            windowed = frame_T[:, ch] * W_long
            frame_F[:, ch] = mdct(windowed)
    
    elif frame_type == 'LONG_START_SEQUENCE' or frame_type == 'LSS':

        # Asymmetric window: [left_Wl/2 (1024), 448 ones, right_Ws/2 (128), 448 zeros]
        window = np.concatenate([
            W_long[:1024],
            np.ones(448),
            W_short[128:], 
            np.zeros(448) 
        ])

        frame_F = np.zeros((1024, 2))
        for ch in range(2):
            windowed = frame_T[:, ch] * window
            frame_F[:, ch] = mdct(windowed)
    
    elif frame_type == 'LONG_STOP_SEQUENCE' or frame_type == 'LPS':

        # Asymmetric window: [448 zeros, left_Ws/2 (128), 448 ones, right_Wl/2 (1024)]
        window = np.concatenate([
            np.zeros(448), 
            W_short[:128], 
            np.ones(448), 
            W_long[1024:] 
        ])

        frame_F = np.zeros((1024, 2))
        for ch in range(2):
            windowed = frame_T[:, ch] * window
            frame_F[:, ch] = mdct(windowed)
    
    elif frame_type == 'EIGHT_SHORT_SEQUENCE' or frame_type == 'ESH':
        
        # Use central 1152 samples, ignore 2x448 samples on sides
        # Divide into 8 overlapping 50% subframes of 256 samples each
        frame_F = np.zeros((1024, 2))  # 8 subframes * 128 coefficients each
        
        for ch in range(2):
            # Extract central 1152 samples
            central_samples = frame_T[448:448+1152, ch]
            
            # Process 8 subframes
            for i in range(8):
                # Extract subframe with 50% overlap
                start = i * 128  # 50% overlap: 256/2 = 128
                subframe = central_samples[start:start+256]
                
                # Apply short window
                windowed = subframe * W_short
                
                # Apply MDCT (256 -> 128 coefficients)
                mdct_coeffs = mdct(windowed)
                
                # Store in frame_F (each subframe contributes 128 coefficients)
                frame_F[i*128:(i+1)*128, ch] = mdct_coeffs
    
    else:
        raise ValueError(f"Unknown frame type: {frame_type}")
    
    return frame_F