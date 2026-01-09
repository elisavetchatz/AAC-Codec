import os
import numpy as np
import soundfile as sf

from aac_coder_2 import aac_coder_2
from i_aac_coder_2 import i_aac_coder_2
from utils_level_1.plotting_utils import (
    plot_audio_waveform,
    plot_audio_spectrogram,
    plot_encoding_process,
    plot_snr_analysis
)
def demo_aac_2(filename_in, filename_out):
    """
    Demonstration of level 2 AAC encoding with TNS support.

    Args:
        filename_in (str): Input .wav file name to be encoded.
        filename_out (str): Output .wav file name where the decoded signal will be stored.
    
    Returns:
        SNR (float): Total Signal-to-Noise Ratio (in dB) between original and decoded signal
    """
    # Read original audio file
    x_original, fs = sf.read(filename_in)
    if fs != 48000:
        raise ValueError("Input must be 48kHz audio")

    # Ensure floating-point representation
    if x_original.dtype == np.int16:
        x_original = x_original.astype(np.float32) / 32768.0
    elif x_original.dtype == np.int32:
        x_original = x_original.astype(np.float32) / 2147483648.0

    # Encode & Decode
    aac_seq_2 = aac_coder_2(filename_in)
    x_decoded = i_aac_coder_2(aac_seq_2, filename_out)

    # Match signal lengths
    if len(x_original) != len(x_decoded):
        min_len = min(len(x_original), len(x_decoded))
        x_original = x_original[:min_len]
        x_decoded = x_decoded[:min_len]
        print(f"Note: Trimmed signals to {min_len} samples for comparison")

    # Compute SNR
    signal_power = np.sum(x_original ** 2)
    noise = x_original - x_decoded
    noise_power = np.sum(noise ** 2)

    if noise_power > 0:
        SNR = 10 * np.log10(signal_power / noise_power)
    else:
        SNR = float('inf')

    # plots
    plot = False
    plot_dir = 'level_2/outputs/plots/sin_window'
    if plot:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"\nGenerating plots in '{plot_dir}/' directory...")

        plot_audio_waveform(
            x_original, x_decoded, fs,
            save_path=f'{plot_dir}/waveform_comparison.png'
        )

        plot_audio_spectrogram(
            x_original, x_decoded, fs,
            save_path=f'{plot_dir}/spectrogram.png'
        )

        plot_encoding_process(
            aac_seq_2, num_frames=10,
            save_path=f'{plot_dir}/encoding_frames.png'
        )

        plot_snr_analysis(
            x_original, x_decoded, fs,
            save_path=f'{plot_dir}/snr_analysis.png'
        )

        print(f"All plots saved! SNR: {SNR:.2f} dB")

    return SNR


if __name__ == "__main__":
    # Create outputs directory and subdirectories
    os.makedirs("level_2/outputs/signals", exist_ok=True)
    
    filename_in = "LicorDeCalandraca.wav"
    filename_out = "level_2/outputs/signals/output_level_2_SIN.wav"
    SNR = demo_aac_2(filename_in, filename_out)
    print(f"SNR between original and decoded signal: {SNR:.2f} dB")