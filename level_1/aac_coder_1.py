def aac_coder_1(filename_in):
    """
    Args:
        filename_in (str): input .wav file name to be encoded
                            Assumption: file includes 2 channel voice with fs = 48kHz
    Returns:
        aac_seq_1 (K length list): K is the number of frams that have been encoded
                                    Each element of this list is a dictionary that includes:
                                    - aac_seq_1[i]['frame_type'] (str): frame type of the i-th frame. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                    - aac_seq_1[i]['win_type'] 
                                    - aac_seq_1[i]['chl']["frame_F"]: MDCT coefficients of left channel (128x8) fr EIGHT SHORT SEQUENCE or (1024x1) for other frame types
                                    - aac_seq_1[i]['chr']["frame_F"]: MDCT coefficients of right channel (128x8) fr EIGHT SHORT SEQUENCE or (1024x1) for other frame types    
    """
    return aac_seq_1