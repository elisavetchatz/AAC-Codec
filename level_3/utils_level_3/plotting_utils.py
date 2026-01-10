import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.gridspec import GridSpec


def plot_audio_waveform(x_original, x_decoded, fs, save_path=None, error_ylim=0.01):
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


def plot_snr_analysis(x_original, x_decoded, fs, save_path=None, frame_size=2048):
    """
    Plot frame-by-frame SNR analysis.
    
    Args:
        x_original (array): Original audio signal
        x_decoded (array): Decoded audio signal
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure
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
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Level 3 AAC: SNR Analysis', fontsize=14, fontweight='bold')
    
    # SNR over time
    axes[0].plot(time_points, snr_values, 'b-', linewidth=1)
    axes[0].set_ylabel('SNR (dB)')
    axes[0].set_title(f'Frame-by-Frame SNR (Frame Size: {frame_size} samples)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(snr_values), color='r', linestyle='--', label=f'Mean SNR: {np.mean(snr_values):.2f} dB')
    axes[0].legend()
    
    # SNR histogram
    axes[1].hist(snr_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Number of Frames')
    axes[1].set_title('SNR Distribution')
    axes[1].axvline(x=np.mean(snr_values), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(snr_values):.2f} dB')
    axes[1].axvline(x=np.median(snr_values), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(snr_values):.2f} dB')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_psychoacoustic_analysis(aac_seq, frame_indices=[0, 50, 100], save_path=None):
    """
    Plot psychoacoustic model analysis for selected frames.
    Shows SMR (Signal-to-Mask Ratio) values per band.
    
    Args:
        aac_seq (list): AAC encoded sequence
        frame_indices (list): Frame indices to plot
        save_path (str, optional): Path to save the figure
    """
    num_plots = len(frame_indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    fig.suptitle('Psychoacoustic Model: Signal-to-Mask Ratio (SMR)', fontsize=14, fontweight='bold')
    
    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(aac_seq):
            continue
            
        frame = aac_seq[frame_idx]
        frame_type = frame['frame_type']
        
        # Get SMR values (stored as T in the frame)
        smr_chl = frame['chl']['T']
        
        if frame_type == 'ESH':
            # For ESH, plot average across subframes
            smr_avg = np.mean(smr_chl, axis=1) if smr_chl.ndim > 1 else smr_chl
            bands = np.arange(len(smr_avg))
            axes[idx].bar(bands, smr_avg, color='blue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Frame {frame_idx} ({frame_type}) - Average SMR across 8 subframes')
        else:
            smr = smr_chl.flatten()
            bands = np.arange(len(smr))
            axes[idx].bar(bands, smr, color='green', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Frame {frame_idx} ({frame_type}) - SMR per Band')
        
        axes[idx].set_xlabel('Scalefactor Band')
        axes[idx].set_ylabel('SMR (dB)')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_xlim([-1, len(bands)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quantization_analysis(aac_seq, frame_indices=[0, 50, 100], save_path=None):
    """
    Plot quantization analysis showing global gain and scalefactor distribution.
    
    Args:
        aac_seq (list): AAC encoded sequence
        frame_indices (list): Frame indices to analyze
        save_path (str, optional): Path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig)
    fig.suptitle('Quantization Analysis: Global Gain and Scalefactors', fontsize=14, fontweight='bold')
    
    # Collect global gain values across all frames
    all_G = []
    for frame in aac_seq:
        G = frame['chl']['G']
        if isinstance(G, np.ndarray):
            if G.ndim > 0:
                all_G.extend(G.flatten())
            else:
                all_G.append(float(G))
        else:
            all_G.append(G)
    
    # Plot 1: Global Gain over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(all_G, 'b-', linewidth=1, alpha=0.7)
    ax1.set_title('Global Gain Evolution Across Frames')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Global Gain (G)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(all_G), color='r', linestyle='--', label=f'Mean: {np.mean(all_G):.2f}')
    ax1.legend()
    
    # Plot 2: Global Gain histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(all_G, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Global Gain')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Global Gain Distribution')
    ax2.axvline(x=np.mean(all_G), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_G):.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Bitstream size distribution
    ax3 = fig.add_subplot(gs[1, 1])
    stream_sizes = [len(frame['chl']['stream']) + len(frame['chr']['stream']) for frame in aac_seq]
    ax3.hist(stream_sizes, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Bitstream Size (bits)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Huffman Encoded Stream Size Distribution')
    ax3.axvline(x=np.mean(stream_sizes), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(stream_sizes):.0f} bits')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Frame type distribution
    ax4 = fig.add_subplot(gs[2, :])
    frame_types = [frame['frame_type'] for frame in aac_seq]
    unique_types, counts = np.unique(frame_types, return_counts=True)
    colors = {'OLS': 'blue', 'LSS': 'green', 'ESH': 'red', 'LPS': 'orange'}
    bar_colors = [colors.get(ft, 'gray') for ft in unique_types]
    
    bars = ax4.bar(unique_types, counts, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Frame Type')
    ax4.set_ylabel('Number of Frames')
    ax4.set_title('Frame Type Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / len(aac_seq)) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_encoding_process(aac_seq, num_frames=3, representative_frames=None, save_path=None):
    """
    Plot MDCT coefficients and quantized symbols for representative frames.
    
    Args:
        aac_seq (list): AAC encoded sequence
        num_frames (int): Number of frames to plot
        representative_frames (list, optional): Specific frame indices to plot
        save_path (str, optional): Path to save the figure
    """
    if representative_frames is None:
        # Evenly spaced frames
        step = len(aac_seq) // (num_frames + 1)
        representative_frames = [step * (i + 1) for i in range(num_frames)]
    
    num_frames = len(representative_frames)
    fig, axes = plt.subplots(num_frames, 2, figsize=(16, 4*num_frames))
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Level 3 Encoding Process: Huffman Bitstreams', fontsize=14, fontweight='bold')
    
    for idx, frame_idx in enumerate(representative_frames):
        if frame_idx >= len(aac_seq):
            continue
            
        frame = aac_seq[frame_idx]
        frame_type = frame['frame_type']
        
        # Left channel bitstream
        stream_chl = frame['chl']['stream']
        sfc_stream_chl = frame['chl']['sfc']
        
        # Convert bitstream to array of bits
        bits_chl = [int(b) for b in stream_chl[:min(500, len(stream_chl))]]  # First 500 bits
        bits_sfc = [int(b) for b in sfc_stream_chl[:min(200, len(sfc_stream_chl))]]  # First 200 bits
        
        # Plot MDCT coefficient bitstream
        axes[idx, 0].step(range(len(bits_chl)), bits_chl, 'b-', linewidth=0.5, where='post')
        axes[idx, 0].fill_between(range(len(bits_chl)), bits_chl, step='post', alpha=0.3)
        axes[idx, 0].set_title(f'Frame {frame_idx} ({frame_type}) - MDCT Coefficients Bitstream\n'
                               f'Total: {len(stream_chl)} bits, Codebook: {frame["chl"]["codebook"]}')
        axes[idx, 0].set_ylabel('Bit Value')
        axes[idx, 0].set_ylim([-0.1, 1.1])
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xlim([0, len(bits_chl)])
        
        # Plot scalefactor bitstream
        axes[idx, 1].step(range(len(bits_sfc)), bits_sfc, 'r-', linewidth=0.5, where='post')
        axes[idx, 1].fill_between(range(len(bits_sfc)), bits_sfc, step='post', alpha=0.3, color='red')
        axes[idx, 1].set_title(f'Frame {frame_idx} ({frame_type}) - Scalefactors Bitstream\n'
                               f'Total: {len(sfc_stream_chl)} bits, Codebook: {frame["chl"]["sfc_codebook"]}')
        axes[idx, 1].set_ylabel('Bit Value')
        axes[idx, 1].set_ylim([-0.1, 1.1])
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim([0, len(bits_sfc)])
        
        if idx == num_frames - 1:
            axes[idx, 0].set_xlabel('Bit Position')
            axes[idx, 1].set_xlabel('Bit Position')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_compression_analysis(aac_seq, fs=48000, save_path=None):
    """
    Plot compression statistics and bit allocation analysis.
    
    Args:
        aac_seq (list): AAC encoded sequence
        fs (int): Sampling frequency
        save_path (str, optional): Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Compression and Bit Allocation Analysis', fontsize=14, fontweight='bold')
    
    # Calculate bits per frame
    bits_per_frame = []
    mdct_bits = []
    sfc_bits = []
    
    for frame in aac_seq:
        total = len(frame['chl']['stream']) + len(frame['chr']['stream'])
        total += len(frame['chl']['sfc']) + len(frame['chr']['sfc'])
        total += 64  # Overhead
        
        bits_per_frame.append(total)
        mdct_bits.append(len(frame['chl']['stream']) + len(frame['chr']['stream']))
        sfc_bits.append(len(frame['chl']['sfc']) + len(frame['chr']['sfc']))
    
    frame_indices = np.arange(len(aac_seq))
    
    # Plot 1: Bits per frame over time
    axes[0, 0].plot(frame_indices, bits_per_frame, 'b-', linewidth=1, label='Total bits')
    axes[0, 0].plot(frame_indices, mdct_bits, 'r-', linewidth=0.7, alpha=0.7, label='MDCT bits')
    axes[0, 0].plot(frame_indices, sfc_bits, 'g-', linewidth=0.7, alpha=0.7, label='Scalefactor bits')
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Bits')
    axes[0, 0].set_title('Bits Allocation Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative bits
    cumulative_bits = np.cumsum(bits_per_frame)
    axes[0, 1].plot(frame_indices, cumulative_bits / 1000, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Cumulative Bits (Kbits)')
    axes[0, 1].set_title('Cumulative Bitstream Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calculate bitrate
    duration = len(aac_seq) * 1024 / fs  # seconds
    avg_bitrate = sum(bits_per_frame) / duration / 1000  # kbps
    axes[0, 1].text(0.05, 0.95, f'Average Bitrate: {avg_bitrate:.2f} kbps',
                    transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Bit distribution pie chart
    total_mdct = sum(mdct_bits)
    total_sfc = sum(sfc_bits)
    total_overhead = len(aac_seq) * 64
    
    sizes = [total_mdct, total_sfc, total_overhead]
    labels = ['MDCT Coefficients', 'Scalefactors', 'Overhead']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    
    axes[1, 0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
    axes[1, 0].set_title('Total Bit Allocation Distribution')
    
    # Plot 4: Compression statistics
    axes[1, 1].axis('off')
    
    # Calculate statistics
    original_bitrate = 16 * 2 * fs / 1000  # kbps (16-bit stereo)
    compression_ratio = original_bitrate / avg_bitrate
    
    stats_text = f"""
    COMPRESSION STATISTICS
    {'='*40}
    
    Original Bitrate:      {original_bitrate:.2f} kbps
    Compressed Bitrate:    {avg_bitrate:.2f} kbps
    Compression Ratio:     {compression_ratio:.2f}x
    
    Total Frames:          {len(aac_seq)}
    Total Bits:            {sum(bits_per_frame):,}
    Average Bits/Frame:    {np.mean(bits_per_frame):.1f}
    
    Bit Distribution:
      - MDCT Coefficients: {total_mdct:,} bits ({100*total_mdct/sum(bits_per_frame):.1f}%)
      - Scalefactors:      {total_sfc:,} bits ({100*total_sfc/sum(bits_per_frame):.1f}%)
      - Overhead:          {total_overhead:,} bits ({100*total_overhead/sum(bits_per_frame):.1f}%)
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
