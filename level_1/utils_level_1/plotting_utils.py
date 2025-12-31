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


def plot_encoding_process(aac_sequence, num_frames=5, save_path=None, representative_frames=None):
    """
    Visualize the encoding process: show MDCT coefficients for actual encoded frames.
    
    Args:
        aac_sequence (list): The AAC sequence output from aac_coder_1
        num_frames (int): Number of frames to visualize
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    # Determine which frames to show
    if representative_frames is not None:
        indices = [i for i in representative_frames if 0 <= i < len(aac_sequence)]
        if len(indices) == 0:
            indices = list(range(min(num_frames, len(aac_sequence))))
    else:
        indices = list(range(min(num_frames, len(aac_sequence))))

    n_frames = len(indices)

    # Collect coefficient arrays (ensure flattened length 1024 per frame)
    coeffs_list = []
    for i in indices:
        frame_data = aac_sequence[i]
        coeffs = frame_data['chl']['frame_F']
        if coeffs.ndim > 1:
            coeffs = coeffs.flatten()
        coeffs_list.append(coeffs)

    # Determine global ylim for fair comparison
    all_vals = np.concatenate([c.flatten() for c in coeffs_list]) if len(coeffs_list) > 0 else np.array([0.0])
    max_abs = np.max(np.abs(all_vals)) if all_vals.size > 0 else 1.0

    # Heatmap mode removed — produce line plots for selected frames only

    # Default: lines per frame
    fig, axes = plt.subplots(n_frames, 1, figsize=(14, 3*n_frames))
    if n_frames == 1:
        axes = [axes]

    fig.suptitle('MDCT Coefficients Per Frame', fontsize=14, fontweight='bold')

    for ax, idx, coeffs in zip(axes, indices, coeffs_list):
        frame_type = aac_sequence[idx]['frame_type']
        ax.plot(coeffs, color='tab:blue', linewidth=0.6)
        ax.set_title(f'Frame {idx+1}: {frame_type}')
        ax.set_ylabel('MDCT Coefficient Value')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-max_abs, max_abs])
        if frame_type == 'ESH':
            for j in range(1, 8):
                ax.axvline(x=j*128, color='r', linestyle='--', alpha=0.5, linewidth=1)

    axes[-1].set_xlabel('Coefficient Index')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Encoding process plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_snr_analysis(x_original, x_decoded, fs, frame_size=2048, save_path=None, energy_threshold=1e-8, snr_clip=(0, 60)):
    """
    Plot SNR over time to show quality variation throughout your audio file.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        frame_size (int): Frame size for SNR calculation
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    # Handle stereo/mono
    if len(x_original.shape) > 1:
        x_orig = x_original[:, 0]
        x_dec = x_decoded[:, 0]
    else:
        x_orig = x_original
        x_dec = x_decoded
    
    # Ensure same length
    min_len = min(len(x_orig), len(x_dec))
    x_orig = x_orig[:min_len]
    x_dec = x_dec[:min_len]
    
    # Calculate frame-by-frame SNR (segmental) with energy threshold
    n_frames = min_len // frame_size
    snr_values = []
    time_points = []
    signal_powers = []
    noise_powers = []
    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        orig_frame = x_orig[start:end]
        dec_frame = x_dec[start:end]
        signal_power = np.sum(orig_frame ** 2)
        noise_power = np.sum((orig_frame - dec_frame) ** 2)
        signal_powers.append(signal_power)
        noise_powers.append(noise_power)
        # Use mean power threshold to exclude silent frames
        mean_power = signal_power / float(frame_size)
        if mean_power >= energy_threshold and noise_power > 0:
            snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
            # clip to reasonable range
            snr = np.clip(snr, snr_clip[0], snr_clip[1])
            snr_values.append(snr)
            time_points.append(start / fs)
    
    # Overall SNR computed over selected (non-silent) frames when possible
    signal_powers = np.array(signal_powers)
    noise_powers = np.array(noise_powers)
    selected = signal_powers / float(frame_size) >= energy_threshold
    if np.any(selected):
        overall_signal_power = np.sum(signal_powers[selected])
        overall_noise_power = np.sum(noise_powers[selected])
    else:
        overall_signal_power = np.sum(x_orig ** 2)
        overall_noise_power = np.sum((x_orig - x_dec) ** 2)
    overall_snr = 10 * np.log10((overall_signal_power + 1e-12) / (overall_noise_power + 1e-12))
    overall_snr = float(np.clip(overall_snr, snr_clip[0], snr_clip[1]))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'SNR Analysis (Overall SNR: {overall_snr:.2f} dB)', fontsize=14, fontweight='bold')
    
    # SNR over time
    axes[0].plot(time_points, snr_values, color='tab:blue', linewidth=1.5)
    axes[0].axhline(y=overall_snr, color='r', linestyle='--', label=f'Overall SNR: {overall_snr:.2f} dB')
    axes[0].set_title('SNR Variation Over Time (segmental)')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Clip y-axis for readability
    axes[0].set_ylim([snr_clip[0], snr_clip[1]])
    
    # SNR histogram
    axes[1].hist(snr_values, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=overall_snr, color='r', linestyle='--', linewidth=2, label=f'Overall SNR: {overall_snr:.2f} dB')
    axes[1].set_title('SNR Distribution (segmental)')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SNR analysis plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return overall_snr
