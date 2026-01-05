import numpy as np

from level_1.utils_level_1.create_kbd_window import create_kbd_window
from level_1.utils_level_1.create_sin_window import create_sin_window
from level_1.utils_level_1.imdct import imdct


def i_filter_bank(frame_F, frame_type, win_type):
    """
    Inverse Filter Bank implementation for a single channel.
    
    Args:
        frame_F: MDCT coefficients for one channel:
                 - (1024, 1) for OLS/LSS/LPS frames
                 - (128, 8) for ESH frames (each column = one subframe)
                 Will be flattened to 1-D internally.
        frame_type: Frame type ('OLS', 'LSS', 'LPS', 'ESH')
        win_type: Window type ('KBD' or 'SIN')
    
    Returns:
        frame_T: 1-D array of 2048 time-domain samples for one channel
    """

    frame_F = np.asarray(frame_F)
    
    # Create windows based on win_type
    if win_type == 'KBD':
        W_long = create_kbd_window(2048, alpha=6)
        W_short = create_kbd_window(256, alpha=4)
    elif win_type == 'SIN':
        W_long = create_sin_window(2048)
        W_short = create_sin_window(256)
    else:
        raise ValueError(f"Unknown window type: {win_type}")
    
    frame_T = np.zeros(2048)
    
    # Process based on frame_type
    if frame_type == 'ONLY_LONG_SEQUENCE' or frame_type == 'OLS':

        # Flatten MDCT coefficients
        frame_F = frame_F.flatten()

        # Use symmetric long window for entire frame
        # Apply IMDCT (1024 -> 2048 samples)
        reconstructed = imdct(frame_F)
        # Apply window in time domain
        frame_T = reconstructed * W_long
    
    elif frame_type == 'LONG_START_SEQUENCE' or frame_type == 'LSS':

        frame_F = frame_F.flatten()

        # Asymmetric window: [left_Wl/2 (1024), 448 ones, right_Ws/2 (128), 448 zeros]
        window = np.concatenate([
            W_long[:1024],           
            np.ones(448),            
            W_short[128:],           
            np.zeros(448)            
        ])

        reconstructed = imdct(frame_F)
        frame_T = reconstructed * window
    
    elif frame_type == 'LONG_STOP_SEQUENCE' or frame_type == 'LPS':

        frame_F = frame_F.flatten()

        # Asymmetric window: [448 zeros, left_Ws/2 (128), 448 ones, right_Wl/2 (1024)]
        window = np.concatenate([
            np.zeros(448),
            W_short[:128], 
            np.ones(448), 
            W_long[1024:]
        ])

        reconstructed = imdct(frame_F)
        frame_T = reconstructed * window
    
    elif frame_type == 'EIGHT_SHORT_SEQUENCE' or frame_type == 'ESH':

        # Reconstruct 8 overlapping subframes for single channel
        # Initialize central samples buffer (1152 samples)
        central_samples = np.zeros(1152)
        
        # Process 8 subframes
        for i in range(8):
            
            # Extract MDCT coefficients for this subframe (128 coefficients)
            mdct_coeffs = frame_F[:, i]
            
            # Apply IMDCT (128 -> 256 samples)
            subframe_reconstructed = imdct(mdct_coeffs)
            
            # Apply short window
            windowed = subframe_reconstructed * W_short
            
            # Overlap-add into central samples
            start = i * 128  # 50% overlap
            central_samples[start:start+256] += windowed
        
        # Place central samples in frame_T with 448 samples padding on each side
        frame_T[448:448+1152] = central_samples
        # The padding regions (first 448 and last 448) remain zero
    
    else:
        raise ValueError(f"Unknown frame type: {frame_type}")
    
    return frame_T
