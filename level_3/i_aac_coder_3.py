def i_aac_coder_3(aac_seq_3, filename_out):
    """
    Inverse AAC coder level 3 implementation (reverses aac_coder_3()).

    Args:
        aac_seq_3 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_3[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_3[i]["win_type"]: Window type
                                   - aac_seq_3[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_3[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_3[i]["chl"]["T"]: Psychoacoustic model thresholds (NB×1) for left channel
                                   - aac_seq_3[i]["chr"]["T"]: Psychoacoustic model thresholds (NB×1) for right channel
                                   - aac_seq_3[i]["chl"]["G"]: Quantized global gains (1 or 8) for left channel
                                   - aac_seq_3[i]["chr"]["G"]: Quantized global gains (1 or 8) for right channel
                                   - aac_seq_3[i]["chl"]["sfc"]: Huffman encoded sequence of sfc for left channel
                                   - aac_seq_3[i]["chr"]["sfc"]: Huffman encoded sequence of sfc for right channel
                                   - aac_seq_3[i]["chl"]["stream"]: Huffman encoded sequence of quantized MDCT 
                                                                    coefficients for left channel
                                   - aac_seq_3[i]["chr"]["stream"]: Huffman encoded sequence of quantized MDCT 
                                                                    coefficients for right channel
                                   - aac_seq_3[i]["chl"]["codebook"]: Huffman codebook used for left channel
                                   - aac_seq_3[i]["chr"]["codebook"]: Huffman codebook used for right channel
        filename_out (str): Output .wav file name where the decoded signal will be stored.
                           Assumption: File will include 2 channel audio with fs = 48kHz.
    
    Returns:
        x (array): Decoded sample sequence
    """
    return x
