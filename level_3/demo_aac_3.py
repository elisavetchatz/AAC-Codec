def demo_aac_3(filename_in, filename_out, filename_aac_coded):
    """
    Demonstration of level 3 AAC encoding with psychoacoustic model and Huffman coding.

    Args:
        filename_in (str): Input .wav file name to be encoded.
        filename_out (str): Output .wav file name where the decoded signal will be stored.
        filename_aac_coded (str): .mat file where the aac_seq_3 structure will be saved.
    
    Returns:
        SNR (float): Total Signal-to-Noise Ratio (in dB) between original and decoded signal
        bitrate (float): Bits per second
        compression (float): Bitrate before encoding divided by bitrate after encoding
    """
    return SNR, bitrate, compression
