import numpy as np
from scipy.io import wavfile

from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1


def demo_acc_1(filename_in, filename_out):

    """
    Showcase of 1st level encoding
    Returns SNR value between original and decoded signal (dB)
    """

    # Read original audio file
    fs, x_original = wavfile.read(filename_in)
    
    # Ensure it's float format for processing
    if x_original.dtype == np.int16:
        x_original = x_original.astype(np.float32) / 32768.0
    elif x_original.dtype == np.int32:
        x_original = x_original.astype(np.float32) / 2147483648.0
    
    # Encode
    aac_seq_1 = aac_coder_1(filename_in)
    
    # Decode
    x_decoded = i_aac_coder_1(aac_seq_1, filename_out)
    
    # Calculate SNR
    # Ensure both signals have the same length
    min_len = min(len(x_original), len(x_decoded))
    x_original = x_original[:min_len]
    x_decoded = x_decoded[:min_len]
    
    # Calculate signal power
    signal_power = np.sum(x_original ** 2)
    
    # Calculate noise (error) power
    noise = x_original - x_decoded
    noise_power = np.sum(noise ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')  # Perfect reconstruction
    
    return SNR