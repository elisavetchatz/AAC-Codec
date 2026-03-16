import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def plot_audio_waveform(x_original, x_decoded, fs, save_path=None, error_ylim=0.01):
    """
    Plot the original vs decoded audio waveform for your specific signal.
    Shows the actual encoding/decoding results.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    # Handle stereo/mono
    if len(x_original.shape) > 1:
        x_orig_plot = x_original[:, 0]  # Left channel
        x_dec_plot = x_decoded[:, 0]
    else:
        x_orig_plot = x_original
        x_dec_plot = x_decoded
    
    # Ensure same length
    min_len = min(len(x_orig_plot), len(x_dec_plot))
    x_orig_plot = x_orig_plot[:min_len]
    x_dec_plot = x_dec_plot[:min_len]
    
    # Time axis in seconds
    time = np.arange(min_len) / fs
    
    # Calculate error
    error = x_orig_plot - x_dec_plot
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('Audio Signal: Original vs Decoded', fontsize=14, fontweight='bold')
    
    # Original signal
    axes[0].plot(time, x_orig_plot, 'b-', linewidth=0.5)
    axes[0].set_title('Original Audio Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time[-1]])
    
    # Decoded signal
    axes[1].plot(time, x_dec_plot, 'r-', linewidth=0.5)
    axes[1].set_title('Decoded Audio Signal (After AAC Encoding/Decoding)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, time[-1]])
    
    # Overlay comparison (zoomed section)
    zoom_samples = min(5000, min_len)
    axes[2].plot(time[:zoom_samples], x_orig_plot[:zoom_samples], 'b-', 
                 linewidth=1, label='Original', alpha=0.7)
    axes[2].plot(time[:zoom_samples], x_dec_plot[:zoom_samples], 'r--', 
                 linewidth=1, label='Decoded', alpha=0.7)
    axes[2].set_title('Overlay Comparison (Zoomed: First 5000 samples)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Error signal (zoomed)
    axes[3].plot(time, error, color='k', linestyle='-', linewidth=0.5)
    axes[3].set_title(f'Reconstruction Error (MSE: {np.mean(error**2):.2e})')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylabel('Error')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, time[-1]])
    # Zoom error y-axis for better visibility
    if error_ylim is not None:
        axes[3].set_ylim([-abs(error_ylim), abs(error_ylim)])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Audio waveform plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_audio_spectrogram(x_original, x_decoded, fs, save_path=None):
    """
    Plot spectrograms of original vs decoded audio.
    Shows frequency content over time for your actual signal.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    # Handle stereo/mono
    if len(x_original.shape) > 1:
        x_orig_plot = x_original[:, 0]
        x_dec_plot = x_decoded[:, 0]
    else:
        x_orig_plot = x_original
        x_dec_plot = x_decoded
    
    # Ensure same length
    min_len = min(len(x_orig_plot), len(x_dec_plot))
    x_orig_plot = x_orig_plot[:min_len]
    x_dec_plot = x_dec_plot[:min_len]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Spectrogram Analysis: Original vs Decoded', fontsize=14, fontweight='bold')

    # Spectrogram parameters
    nperseg = 2048
    noverlap = 1024

    f, t, Sxx_orig = signal.spectrogram(
        x_orig_plot,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx_orig + 1e-10), shading='gouraud', cmap='viridis')
    axes[0].set_title('Original Signal Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    # Decoded spectrogram (SciPy)
    f, t, Sxx_dec = signal.spectrogram(
        x_dec_plot,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_dec + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_title('Decoded Signal Spectrogram')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')
    # Annotate STFT params on decoded subplot
    axes[1].text(0.01, 0.95, f'nperseg={nperseg}, noverlap={noverlap}', transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
