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

    # Detect short vs long blocks by shape rather than solely by frame_type
    if frame_F_in.ndim == 1:
        n0 = frame_F_in.shape[0]
        if n0 == 1024:
            is_short = False
        elif n0 == 128:
            frame_F_in = frame_F_in.reshape(128, 1)
            is_short = True
        else:
            is_short = frame_type in ('ESH', 'EIGHT_SHORT_SEQUENCE')
    else:
        rows = frame_F_in.shape[0]
        if rows == 128:
            is_short = True
        elif rows == 1024:
            is_short = False
        else:
            is_short = frame_type in ('ESH', 'EIGHT_SHORT_SEQUENCE')

    if is_short:
        # Handle any number of short subframes
        n_subframes = frame_F_in.shape[1]
        frame_F_out = np.zeros_like(frame_F_in)

        # Normalize tns_coeffs shape to (4, n_subframes)
        if tns_coeffs.ndim == 1:
            tns_coeffs = tns_coeffs.reshape(-1, 1)
        if tns_coeffs.shape[1] != n_subframes:
            # Try to broadcast/reshape if possible
            if tns_coeffs.shape[1] == 1:
                tns_coeffs = np.tile(tns_coeffs, (1, n_subframes))
            else:
                # Mismatch — continue but will pick columns safely
                pass

        for subframe in range(n_subframes):
            Y = frame_F_in[:, subframe]
            # Safely pick coefficients column (or last column if missing)
            if tns_coeffs.shape[1] > subframe:
                a = tns_coeffs[:, subframe]
            else:
                a = tns_coeffs[:, -1]

            # Apply inverse TNS filter
            X = apply_inverse_tns_filter(Y, a)

            X = np.asarray(X).flatten()
            # Defensive resizing if lengths differ
            if X.shape[0] != frame_F_out.shape[0]:
                if X.shape[0] > frame_F_out.shape[0]:
                    X = X[: frame_F_out.shape[0]]
                else:
                    X = np.pad(X, (0, frame_F_out.shape[0] - X.shape[0]))

            frame_F_out[:, subframe] = X

    else:
        # Long frame
        Y = frame_F_in.flatten()
        a = tns_coeffs.flatten()

        # Apply inverse TNS filter
        X = apply_inverse_tns_filter(Y, a)

        X = np.asarray(X).flatten()
        frame_F_out = X.reshape(-1, 1)
    
    return frame_F_out