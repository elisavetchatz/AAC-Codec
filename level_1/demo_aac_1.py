import os
import numpy as np
import soundfile as sf

from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1
from utils_level_1.plotting_utils import (plot_audio_waveform, plot_audio_spectrogram)

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
    eps = 1e-12
    SNR = 10 * np.log10(signal_power / (noise_power + eps))

    
    # Generate plots
    plot = False
    plot_dir = 'level_1/outputs/plots/sin_window'
    if plot:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"\nGenerating plots in '{plot_dir}/' directory...")

        # Waveform comparison (zoom error with default ±0.01)
        plot_audio_waveform(x_original, x_decoded, fs, save_path=f'{plot_dir}/waveform_comparison.png')

        # Spectrogram analysis (original vs decoded)
        plot_audio_spectrogram(x_original, x_decoded, fs, save_path=f'{plot_dir}/spectrogram.png')

        print(f"All plots saved! SNR: {SNR:.2f} dB")

    return SNR

if __name__ == "__main__":
    # Create outputs directory and subdirectories
    os.makedirs("level_1/outputs/signals", exist_ok=True)
    
    filename_in = "LicorDeCalandraca.wav"
    filename_out = "level_1/outputs/signals/output_level_1_SIN.wav"
    SNR = demo_acc_1(filename_in, filename_out)
    print(f"SNR between original and decoded signal: {SNR:.2f} dB")