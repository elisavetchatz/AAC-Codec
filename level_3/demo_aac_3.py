import os
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
    
    # Calculate bitrate and analyze sparsity
    total_stream_bits = 0
    total_sfc_bits = 0
    total_overhead_bits = 0
    total_nonzero_coeffs = 0
    total_coeffs = 0
    
    for frame in aac_seq_3:
        # Handle both string and array formats for streams
        chl_stream = frame["chl"]["stream"]
        chl_sfc = frame["chl"]["sfc"]
        chr_stream = frame["chr"]["stream"]
        chr_sfc = frame["chr"]["sfc"]
        
        # Convert to string if necessary (from .mat file loading)
        if isinstance(chl_stream, np.ndarray):
            chl_stream = ''.join(chl_stream.flatten().astype(str))
        if isinstance(chl_sfc, np.ndarray):
            chl_sfc = ''.join(chl_sfc.flatten().astype(str))
        if isinstance(chr_stream, np.ndarray):
            chr_stream = ''.join(chr_stream.flatten().astype(str))
        if isinstance(chr_sfc, np.ndarray):
            chr_sfc = ''.join(chr_sfc.flatten().astype(str))
        
        total_stream_bits += len(chl_stream) + len(chr_stream)
        total_sfc_bits += len(chl_sfc) + len(chr_sfc)
        total_overhead_bits += 64  # Frame header, etc.
        
        # Count non-zero coefficients
        if "chl" in frame and "nonzero_coeffs" in frame["chl"]:
            total_nonzero_coeffs += frame["chl"]["nonzero_coeffs"]
        if "chr" in frame and "nonzero_coeffs" in frame["chr"]:
            total_nonzero_coeffs += frame["chr"]["nonzero_coeffs"]
        if "total_coeffs" in frame:
            total_coeffs += frame["total_coeffs"] * 2  # *2 for both channels
    
    total_bits = total_stream_bits + total_sfc_bits + total_overhead_bits
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
    print(f"Bitrate breakdown:")
    print(f"  Stream bits: {total_stream_bits:,} ({total_stream_bits/total_bits*100:.1f}%)")
    print(f"  SFC bits: {total_sfc_bits:,} ({total_sfc_bits/total_bits*100:.1f}%)")
    print(f"  Overhead bits: {total_overhead_bits:,} ({total_overhead_bits/total_bits*100:.1f}%)")
    print("=" * 70)
    if total_coeffs > 0:
        print(f"Coefficient sparsity:")
        print(f"  Non-zero coefficients: {total_nonzero_coeffs:,} / {total_coeffs:,} ({total_nonzero_coeffs/total_coeffs*100:.1f}%)")
        print(f"  Zero coefficients: {total_coeffs - total_nonzero_coeffs:,} ({(total_coeffs - total_nonzero_coeffs)/total_coeffs*100:.1f}%)")
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

