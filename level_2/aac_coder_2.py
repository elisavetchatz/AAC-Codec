def aac_coder_2(filename_in):
    """
    AAC coder level 2 implementation with TNS support.

    Args:
        filename_in (str): Input .wav file name to be encoded.
                          Assumption: File includes 2 channel audio with fs = 48kHz.
    
    Returns:
        aac_seq_2 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_2[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_2[i]["win_type"]: Window type
                                   - aac_seq_2[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_2[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_2[i]["chl"]["frame_F"]: MDCT coefficients for left channel after TNS
                                   - aac_seq_2[i]["chr"]["frame_F"]: MDCT coefficients for right channel after TNS
    """
    return aac_seq_2
