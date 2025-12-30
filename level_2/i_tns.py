import numpy as np
from utils_level_2.filter_utils import apply_inverse_tns_filter

def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Inverse Temporal Noise Shaping (TNS) stage implementation.

    Args:
        frame_F_in (array): MDCT coefficients before inverse TNS
                           Dimensions: 128x8 for EIGHT_SHORT_SEQUENCE, 1024x1 otherwise
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
        tns_coeffs (array): Quantized TNS coefficients
                           Dimensions: 4x8 for EIGHT_SHORT_SEQUENCE, 4x1 otherwise
    
    Returns:
        frame_F_out (array): MDCT coefficients after inverse TNS
                            Dimensions: 128x8 for EIGHT_SHORT_SEQUENCE, 1024x1 otherwise
    """
    frame_F_in = np.asarray(frame_F_in)
    tns_coeffs = np.asarray(tns_coeffs)
    
    if frame_type == 'ESH' or frame_type == 'EIGHT_SHORT_SEQUENCE':

        # Process 8 short subframes
        frame_F_out = np.zeros((128, 8))
        
        for subframe in range(8):

            # Extract subframe coefficients and TNS coefficients
            Y = frame_F_in[:, subframe]
            a = tns_coeffs[:, subframe]
            
            # Apply inverse TNS filter
            X = apply_inverse_tns_filter(Y, a)
            
            frame_F_out[:, subframe] = X
    
    else:

        # Process single long frame
        Y = frame_F_in.flatten()
        a = tns_coeffs.flatten()
        
        # Apply inverse TNS filter
        X = apply_inverse_tns_filter(Y, a)
        
        frame_F_out = X.reshape(-1, 1)
    
    return frame_F_out