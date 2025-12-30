import os
import numpy as np
import soundfile as sf

from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1
from utils_level_1.plotting_utils import (plot_audio_waveform, plot_audio_spectrogram,
                                          plot_encoding_process, plot_snr_analysis)

def demo_acc_1(filename_in, filename_out):

    """
    Showcase of 1st level encoding
    Returns SNR value between original and decoded signal (dB)
    """

    # Read original audio file
    x_original, fs = sf.read(filename_in)
    
    # Ensure it's float format for processing
    if x_original.dtype == np.int16:
        x_original = x_original.astype(np.float32) / 32768.0
    elif x_original.dtype == np.int32:
        x_original = x_original.astype(np.float32) / 2147483648.0
    
    # Encode
    aac_seq_1 = aac_coder_1(filename_in)
    # Decode
    x_decoded = i_aac_coder_1(aac_seq_1, filename_out)
    print("originl len:", len(x_original))
    print("decoded len:", len(x_decoded))
    
    # Ensure both signals have the same length
    if len(x_original) != len(x_decoded):
        min_len = min(len(x_original), len(x_decoded))
        x_original = x_original[:min_len]
        x_decoded = x_decoded[:min_len]
        print(f"Note: Trimmed signals to {min_len} samples for comparison")
    
    # Calculate signal power
    signal_power = np.sum(x_original ** 2)
    
    # Calculate noise
    noise = x_original - x_decoded
    noise_power = np.sum(noise ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')  # Perfect reconstruction
    
    # Generate plots if requested
    plot = True
    plot_dir = 'level_1/plots'
    if plot:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"\nGenerating plots in '{plot_dir}/' directory...")
        
        # Waveform comparison
        plot_audio_waveform(x_original, x_decoded, fs, 
                           save_path=f'{plot_dir}/waveform_comparison.png')
        # Spectrogram analysis
        plot_audio_spectrogram(x_original, x_decoded, fs, 
                              save_path=f'{plot_dir}/spectrogram.png')
        # Encoding (first 10 frames)
        plot_encoding_process(aac_seq_1, num_frames=10, 
                             save_path=f'{plot_dir}/encoding_frames.png')
        # SNR
        plot_snr_analysis(x_original, x_decoded, fs, 
                         save_path=f'{plot_dir}/snr_analysis.png')
        print(f"All plots saved! SNR: {SNR:.2f} dB")

    return SNR

if __name__ == "__main__":
    filename_in = "LicorDeCalandraca.wav"
    filename_out = "output.wav"
    SNR = demo_acc_1(filename_in, filename_out)
    print(f"SNR between original and decoded signal: {SNR:.2f} dB")