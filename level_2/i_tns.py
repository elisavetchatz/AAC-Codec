
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
    return frame_F_out