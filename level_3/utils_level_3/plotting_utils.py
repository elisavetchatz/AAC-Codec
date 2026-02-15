import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.gridspec import GridSpec


def plot_audio_waveform(x_original, x_decoded, fs, save_path=None, error_ylim=0.5):
    """
    Plot the original vs decoded audio waveform.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure
        error_ylim (float): Y-axis limit for error plot
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
    fig.suptitle('Level 3 AAC: Audio Signal Analysis', fontsize=14, fontweight='bold')
    
    # Original signal
    axes[0].plot(time, x_orig_plot, 'b-', linewidth=0.5)
    axes[0].set_title('Original Audio Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time[-1]])
    
    # Decoded signal
    axes[1].plot(time, x_dec_plot, 'r-', linewidth=0.5)
    axes[1].set_title('Decoded Audio Signal (After Psychoacoustic Encoding)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, time[-1]])
    
    # Error
    axes[2].plot(time, error, 'g-', linewidth=0.5)
    axes[2].set_title('Reconstruction Error')
    axes[2].set_ylabel('Error')
    axes[2].set_ylim([-error_ylim, error_ylim])
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, time[-1]])
    
    # Zoom on a small segment
    zoom_start = int(0.1 * min_len)
    zoom_end = int(0.11 * min_len)
    time_zoom = time[zoom_start:zoom_end]
    
    axes[3].plot(time_zoom, x_orig_plot[zoom_start:zoom_end], 'b-', label='Original', linewidth=1)
    axes[3].plot(time_zoom, x_dec_plot[zoom_start:zoom_end], 'r--', label='Decoded', linewidth=1)
    axes[3].set_title('Zoomed Comparison (10%-11% of signal)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_audio_spectrogram(x_original, x_decoded, fs, save_path=None):
    """
    Plot spectrograms of original and decoded signals.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure
    """
    # Handle stereo/mono
    if len(x_original.shape) > 1:
        x_orig = x_original[:, 0]
        x_dec = x_decoded[:, 0]
    else:
        x_orig = x_original
        x_dec = x_decoded
    
    min_len = min(len(x_orig), len(x_dec))
    x_orig = x_orig[:min_len]
    x_dec = x_dec[:min_len]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Level 3 AAC: Spectrogram Analysis', fontsize=14, fontweight='bold')
    
    # Original spectrogram
    f, t, Sxx_orig = signal.spectrogram(x_orig, fs, nperseg=2048, noverlap=1024)
    im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx_orig + 1e-10), shading='gouraud', cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Original Signal Spectrogram')
    axes[0].set_ylim([0, fs/2])
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')
    
    # Decoded spectrogram
    f, t, Sxx_dec = signal.spectrogram(x_dec, fs, nperseg=2048, noverlap=1024)
    im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_dec + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Decoded Signal Spectrogram')
    axes[1].set_ylim([0, fs/2])
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')
    
    # Difference
    diff = 10 * np.log10(Sxx_orig + 1e-10) - 10 * np.log10(Sxx_dec + 1e-10)
    im3 = axes[2].pcolormesh(t, f, diff, shading='gouraud', cmap='RdBu_r', vmin=-10, vmax=10)
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Spectral Difference (Original - Decoded)')
    axes[2].set_ylim([0, fs/2])
    plt.colorbar(im3, ax=axes[2], label='Difference (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_snr_analysis(x_original, x_decoded, fs, save_dir=None, frame_size=2048):
    """
    Plot frame-by-frame SNR analysis in separate figures.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_dir (str, optional): Directory to save the figures
        frame_size (int): Frame size for SNR calculation
    """
    # Handle stereo/mono
    if len(x_original.shape) > 1:
        x_orig = x_original[:, 0]
        x_dec = x_decoded[:, 0]
    else:
        x_orig = x_original
        x_dec = x_decoded
    
    min_len = min(len(x_orig), len(x_dec))
    x_orig = x_orig[:min_len]
    x_dec = x_dec[:min_len]
    
    # Calculate global SNR
    global_signal_power = np.sum(x_orig ** 2)
    global_noise_power = np.sum((x_orig - x_dec) ** 2)
    if global_noise_power > 0:
        global_snr = 10 * np.log10(global_signal_power / global_noise_power)
    else:
        global_snr = float('inf')
    
    # Calculate frame-by-frame SNR
    num_frames = min_len // frame_size
    snr_values = []
    time_points = []
    
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        
        orig_frame = x_orig[start:end]
        dec_frame = x_dec[start:end]
        
        signal_power = np.sum(orig_frame ** 2)
        noise_power = np.sum((orig_frame - dec_frame) ** 2)
        
        if noise_power > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr)
            time_points.append(start / fs)
    
    # ========== Figure 1: SNR over time ==========
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    
    ax1.plot(time_points, snr_values, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Frame-by-Frame SNR (Frame Size: {frame_size} samples)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(snr_values), color='r', linestyle='--', linewidth=2, 
                label=f'Mean Frame SNR: {np.mean(snr_values):.2f} dB')
    ax1.axhline(y=global_snr, color='orange', linestyle=':', linewidth=2, 
                label=f'Global SNR: {global_snr:.2f} dB')
    ax1.legend()
    
    fig1.tight_layout()
    if save_dir:
        save_path1 = f'{save_dir}/snr_over_time.png'
        fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
    
    # ========== Figure 2: SNR distribution ==========
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    
    ax2.hist(snr_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Frames', fontsize=11, fontweight='bold')
    ax2.set_title('SNR Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(x=np.mean(snr_values), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(snr_values):.2f} dB')
    ax2.axvline(x=np.median(snr_values), color='g', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(snr_values):.2f} dB')
    ax2.axvline(x=global_snr, color='orange', linestyle=':', linewidth=2, 
                label=f'Global: {global_snr:.2f} dB')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig2.tight_layout()
    if save_dir:
        save_path2 = f'{save_dir}/snr_distribution.png'
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    if not save_dir:
        plt.show()




def plot_compression_analysis(aac_seq, fs=48000, save_dir=None):
    """
    Plot PQMF analysis and bit allocation, print compression statistics.
    
    Args:
        aac_seq (list): AAC encoded sequence
        fs (int): Sampling frequency
        save_dir (str, optional): Directory to save the figures
    """
    
    # Calculate bits per frame
    bits_per_frame = []
    mdct_bits = []
    sfc_bits = []
    
    for frame in aac_seq:
        # Handle both string and array formats for streams
        chl_stream = frame["chl"]["stream"]
        chl_sfc = frame["chl"]["sfc"]
        chr_stream = frame["chr"]["stream"]
        chr_sfc = frame["chr"]["sfc"]
        
        # Convert to string if necessary
        if isinstance(chl_stream, np.ndarray):
            chl_stream = ''.join(chl_stream.flatten().astype(str))
        if isinstance(chl_sfc, np.ndarray):
            chl_sfc = ''.join(chl_sfc.flatten().astype(str))
        if isinstance(chr_stream, np.ndarray):
            chr_stream = ''.join(chr_stream.flatten().astype(str))
        if isinstance(chr_sfc, np.ndarray):
            chr_sfc = ''.join(chr_sfc.flatten().astype(str))
        
        stream_bits = len(chl_stream) + len(chr_stream)
        sf_bits = len(chl_sfc) + len(chr_sfc)
        overhead = 64
        
        bits_per_frame.append(stream_bits + sf_bits + overhead)
        mdct_bits.append(stream_bits)
        sfc_bits.append(sf_bits)
    
    frame_indices = np.arange(len(aac_seq))
    
    # ========== Figure 1: PQMF Analysis (Bits allocation over time) ==========
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    
    ax1.plot(frame_indices, bits_per_frame, 'b-', linewidth=1.5, label='Total bits')
    ax1.plot(frame_indices, mdct_bits, 'r-', linewidth=1, alpha=0.7, label='MDCT bits')
    ax1.plot(frame_indices, sfc_bits, 'g-', linewidth=1, alpha=0.7, label='Scalefactor bits')
    ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bits', fontsize=11, fontweight='bold')
    ax1.set_title('Bits Allocation Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    if save_dir:
        save_path1 = f'{save_dir}/pqmf_analysis.png'
        fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
    
    # ========== Figure 2: Bit Allocation Distribution ==========
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    total_stream_bits = sum(mdct_bits)
    total_sfc_bits = sum(sfc_bits)
    total_overhead_bits = len(aac_seq) * 64
    total_bits = total_stream_bits + total_sfc_bits + total_overhead_bits
    
    sizes = [total_stream_bits, total_sfc_bits, total_overhead_bits]
    labels = ['MDCT Coefficients', 'Scalefactors', 'Overhead']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.08, 0.08, 0.08)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Total Bit Allocation Distribution', fontsize=12, fontweight='bold')
    
    fig2.tight_layout()
    if save_dir:
        save_path2 = f'{save_dir}/bit_allocation.png'
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # ========== Print Compression Statistics ==========
    duration = len(aac_seq) * 1024 / fs  # seconds
    avg_bitrate = sum(bits_per_frame) / duration / 1000  # kbps
    original_bitrate = 16 * 2 * fs / 1000  # kbps (16-bit stereo)
    compression_ratio = original_bitrate / avg_bitrate
    
    print(f"\n{'='*60}")
    print("COMPRESSION STATISTICS")
    print(f"{'='*60}")
    print(f"\nOriginal Bitrate:      {original_bitrate:.2f} kbps")
    print(f"Compressed Bitrate:    {avg_bitrate:.2f} kbps")
    print(f"Compression Ratio:     {compression_ratio:.2f}x")
    print(f"\nTotal Frames:          {len(aac_seq)}")
    print(f"Total Bits:            {total_bits:,}")
    print(f"Average Bits/Frame:    {np.mean(bits_per_frame):.1f}")
    print(f"\nBitrate Breakdown:")
    print(f"  - Stream bits:       {total_stream_bits:,} bits ({100*total_stream_bits/total_bits:.1f}%)")
    print(f"  - SFC bits:          {total_sfc_bits:,} bits ({100*total_sfc_bits/total_bits:.1f}%)")
    print(f"  - Overhead bits:     {total_overhead_bits:,} bits ({100*total_overhead_bits/total_bits:.1f}%)")
    print(f"{'='*60}\n")
    
    if not save_dir:
        plt.show()
