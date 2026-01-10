import numpy as np

from utils_level_2.tns_utils import compute_normalization_factors, solve_lpc_coeffs, quantize_tns_coeffs
from utils_level_2.filter_utils import apply_tns_filter, check_filter_stability

def tns(frame_F_in, frame_type):
    """
    Temporal Noise Shaping (TNS) stage implementation for one channel.

    Args:
        frame_F_in (array): MDCT coefficients before Temporal Noise Shaping
                           Dimensions: 128x8 for EIGHT_SHORT_SEQUENCE, 1024x1 otherwise
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
    
    Returns:
        frame_F_out (array): MDCT coefficients after Temporal Noise Shaping
                            Dimensions: 128x8 for EIGHT_SHORT_SEQUENCE, 1024x1 otherwise
        tns_coeffs (array): Quantized TNS coefficients
                           Dimensions: 4x8 for EIGHT_SHORT_SEQUENCE, 4x1 otherwise
    """
    frame_F_in = np.asarray(frame_F_in)
    
    if frame_type == 'ESH' or frame_type == 'EIGHT_SHORT_SEQUENCE':

        # Process 8 short subframes
        frame_F_out = np.zeros((128, 8))
        tns_coeffs = np.zeros((4, 8))
        
        for subframe in range(8):

            # Extract subframe coefficients
            X = frame_F_in[:, subframe]
            
            # Normalize coefficients
            S_w = compute_normalization_factors(X, 'ESH')
            X_w = X / (S_w + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Compute LPC coefficients
            a = solve_lpc_coeffs(X_w, order=4)
            
            # Quantize coefficients
            a_quant = quantize_tns_coeffs(a)
            
            # Check stability
            if not check_filter_stability(a_quant):

                # If unstable, use zero coefficients (no TNS)
                print("Unstable TNS filter detected in subframe {}, using zero coefficients.".format(subframe))
                a_quant = np.zeros(4)
            
            # Apply TNS filter
            Y = apply_tns_filter(X, a_quant)
            
            frame_F_out[:, subframe] = Y
            tns_coeffs[:, subframe] = a_quant
    
    else:

        # Process single long frame
        X = frame_F_in.flatten()
        
        # Normalize coefficients
        S_w = compute_normalization_factors(X, frame_type)
        X_w = X / (S_w + 1e-10)
        
        # Compute LPC coefficients
        a = solve_lpc_coeffs(X_w, order=4)
        
        # Quantize coefficients
        a_quant = quantize_tns_coeffs(a)
        
        # Check stability
        if not check_filter_stability(a_quant):
            # If unstable, use zero coefficients (no TNS)
            print("Unstable TNS filter detected in long frame, using zero coefficients.")
            a_quant = np.zeros(4)
        
        # Apply TNS filter
        Y = apply_tns_filter(X, a_quant)
        
        frame_F_out = Y.reshape(-1, 1)
        tns_coeffs = a_quant.reshape(-1, 1)
    
    return frame_F_out, tns_coeffs