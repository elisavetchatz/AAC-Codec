import os
import sys
import numpy as np
import soundfile as sf
import scipy.io as sio

from aac_coder_3 import aac_coder_3
from i_aac_coder_3 import i_aac_coder_3


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

    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    x_decoded = i_aac_coder_3(aac_seq_3, filename_out)
    
    x_original, fs = sf.read(filename_in)
    
    min_len = min(len(x_original), len(x_decoded))
    x_original = x_original[:min_len]
    x_decoded = x_decoded[:min_len]
    
    # SNR
    signal_power = np.sum(x_original ** 2)
    noise_power = np.sum((x_original - x_decoded) ** 2)
    
    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')
    
    # Calculate bitrate
    total_bits = 0
    for frame in aac_seq_3:

        total_bits += len(frame["chl"]["stream"])
        total_bits += len(frame["chl"]["sfc"])
        
        total_bits += len(frame["chr"]["stream"])
        total_bits += len(frame["chr"]["sfc"])
        
        total_bits += 64
    
    duration = len(x_original) / fs
    bitrate = total_bits / duration
    
    original_bitrate = 16 * 2 * fs
    compression = original_bitrate / bitrate
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"SNR: {SNR:.2f} dB")
    print(f"Bitrate: {bitrate:.2f} bits/second ({bitrate/1000:.2f} kbps)")
    print(f"Original bitrate: {original_bitrate/1000:.2f} kbps")
    print(f"Compression ratio: {compression:.2f}x")
    print(f"Number of frames: {len(aac_seq_3)}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 70)
    
    return SNR, bitrate, compression


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    filename_in = os.path.join(root_dir, "LicorDeCalandraca.wav")
    filename_out = os.path.join(current_dir, "outputs", "signals", "output_level_3.wav")
    filename_aac_coded = os.path.join(current_dir, "outputs", "aac_seq_3.mat")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    os.makedirs(os.path.dirname(filename_aac_coded), exist_ok=True)
    
    SNR, bitrate, compression = demo_aac_3(filename_in, filename_out, filename_aac_coded)

