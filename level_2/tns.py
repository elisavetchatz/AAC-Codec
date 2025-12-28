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
    return [frame_F_out, tns_coeffs]