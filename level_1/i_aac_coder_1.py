def i_aac_coder_1(aac_seq_1, filename_out):
    """
    Inverse AAC coder aac_coder_1() implementation.

    Args:
        aac_seq_1 (K length list)
        filename_out (str): output .wav file name where the decoded signal will be stored
                            Assumption: file includes 2 channel voice with fs = 48kHz
    Returns:
        x : decoded sample sequence
    """
    return x