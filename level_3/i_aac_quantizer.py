def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse quantizer stage implementation (reverses aac_quantizer()).

    Args:
        S (array): Array of quantized MDCT coefficient symbols for current frame
                  Dimensions: 1024×1 (for all frame types)
        sfc (array): Scalefactor values for each Scalefactor band
                    Dimensions: NB×8 for EIGHT_SHORT_SEQUENCE frames (where NB is number of bands),
                               NB×1 for all other types
        G (array or float): Global gain of current frame
                           Dimensions: 1×8 for EIGHT_SHORT_SEQUENCE or single value for all other cases
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
    
    Returns:
        frame_F (array): MDCT coefficients
                        Dimensions: 128×8 for EIGHT_SHORT_SEQUENCE, 1024×1 otherwise
    """
    return frame_F
