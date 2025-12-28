def i_aac_coder_2(aac_seq_2, filename_out):
    """
    Inverse AAC coder level 2 implementation (reverses aac_coder_2()).

    Args:
        aac_seq_2 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_2[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_2[i]["win_type"]: Window type
                                   - aac_seq_2[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_2[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_2[i]["chl"]["frame_F"]: MDCT coefficients for left channel after TNS
                                   - aac_seq_2[i]["chr"]["frame_F"]: MDCT coefficients for right channel after TNS
        filename_out (str): Output .wav file name where the decoded signal will be stored.
                           Assumption: File will include 2 channel audio with fs = 48kHz.
    
    Returns:
        x (array): Decoded sample sequence
    """
    return x
