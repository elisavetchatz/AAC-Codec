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

    # Determine whether the input corresponds to short blocks or a long block
    # Prefer shape-based detection to avoid mismatches between `frame_type`
    # and the actual array dimensions coming from upstream code.
    if frame_F_in.ndim == 1:
        n0 = frame_F_in.shape[0]
        if n0 == 1024:
            is_short = False
        elif n0 == 128:
            # Single short subframe provided as 1D array
            frame_F_in = frame_F_in.reshape(128, 1)
            is_short = True
        else:
            is_short = frame_type in ('ESH', 'EIGHT_SHORT_SEQUENCE')
    else:
        # 2D input: decide by number of rows
        rows = frame_F_in.shape[0]
        if rows == 128:
            is_short = True
        elif rows == 1024:
            is_short = False
        else:
            is_short = frame_type in ('ESH', 'EIGHT_SHORT_SEQUENCE')

    if is_short:

        # Process short subframes (handle any number of subframes)
        n_subframes = frame_F_in.shape[1]
        frame_F_out = np.zeros_like(frame_F_in)
        tns_coeffs = np.zeros((4, n_subframes))
        
        for subframe in range(n_subframes):

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
            
            # Ensure output length matches input subframe length
            Y = np.asarray(Y).flatten()
            if Y.shape[0] != frame_F_out.shape[0]:
                # If lengths mismatch, try to adapt: if Y is long, trim; if short, pad with zeros
                if Y.shape[0] > frame_F_out.shape[0]:
                    Y = Y[: frame_F_out.shape[0]]
                else:
                    Y = np.pad(Y, (0, frame_F_out.shape[0] - Y.shape[0]))

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

        Y = np.asarray(Y).flatten()
        frame_F_out = Y.reshape(-1, 1)
        tns_coeffs = a_quant.reshape(-1, 1)
    
    return frame_F_out, tns_coeffs