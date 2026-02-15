import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import scipy.io as sio


# Codebook tuple sizes (AAC standard)
CODEBOOK_TUPLE_SIZE = {
    1: 4, 2: 4,  # 4-tuples, signed
    3: 4, 4: 4,  # 4-tuples, unsigned
    5: 2, 6: 2,  # 2-tuples, signed
    7: 2, 8: 2,  # 2-tuples, unsigned
    9: 2, 10: 2, 11: 2  # 2-tuples, unsigned
}


def compute_entropy(elements):
    """
    Compute Shannon entropy of a sequence.
    
    H(X) = -Σ p(x) · log₂(p(x))
    
    Parameters
    ----------
    elements : list or array
        Sequence of symbols or tuples
    
    Returns
    -------
    entropy : float
        Shannon entropy in bits per element
    probabilities : dict
        Probability distribution {element: probability}
    """
    if len(elements) == 0:
        return 0.0, {}
    
    # Count frequencies
    counts = Counter(elements)
    total = len(elements)
    
    # Compute probabilities
    probabilities = {elem: count / total for elem, count in counts.items()}
    
    # Compute entropy
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    return entropy, probabilities


def extract_tuples(symbols, codebook_id):
    """
    Group symbols into tuples based on AAC codebook specification.
    
    Parameters
    ----------
    symbols : array_like
        Quantized MDCT coefficients
    codebook_id : int
        Huffman codebook used (1-11)
    
    Returns
    -------
    tuples : list of tuples
        Grouped symbols
    """
    symbols = np.asarray(symbols).flatten()
    
    if codebook_id == 0:  # All zeros
        return []
    
    tuple_size = CODEBOOK_TUPLE_SIZE.get(codebook_id, 2)
    num_tuples = len(symbols) // tuple_size
    
    tuples = []
    for i in range(num_tuples):
        start = i * tuple_size
        end = start + tuple_size
        tuples.append(tuple(symbols[start:end]))
    
    return tuples


def analyze_frame_entropy(frame_data, frame_idx=0, verbose=False, channel='left'):
    """
    Analyze entropy efficiency for a single AAC frame.
    
    Parameters
    ----------
    frame_data : dict or structured array
        AAC frame from aac_seq_3
    frame_idx : int
        Frame index for reporting
    verbose : bool
        Print detailed analysis
    channel : str
        'left', 'right', or 'both' - which channel to analyze
    
    Returns
    -------
    results : dict
        Analysis results including H, L, efficiency, redundancy
    """
    # Extract channel data - handle the nested structure
    if channel == 'left' or channel == 'both':
        ch_data = frame_data['chl'][0, 0] if isinstance(frame_data['chl'], np.ndarray) else frame_data['chl']
    else:
        ch_data = frame_data['chr'][0, 0] if isinstance(frame_data['chr'], np.ndarray) else frame_data['chr']
    
    # Extract bitstream
    stream_raw = ch_data['stream']
    if isinstance(stream_raw, np.ndarray):
        # Extract from numpy array
        bitstream = stream_raw[0, 0] if stream_raw.size == 1 else stream_raw[0]
        if isinstance(bitstream, np.ndarray):
            bitstream = bitstream[0] if bitstream.size == 1 else ''.join(str(b) for b in bitstream.flatten())
    else:
        bitstream = str(stream_raw)
    
    
    # Extract codebook info
    codebook_raw = ch_data['codebook']
    if isinstance(codebook_raw, np.ndarray):
        codebook_arr = codebook_raw.flatten()
        # Get most common codebook
        codebook = int(np.median(codebook_arr[codebook_arr > 0]).item()) if len(codebook_arr[codebook_arr > 0]) > 0 else 3
    else:
        codebook = int(codebook_raw)
    
    # Try to extract quantized symbols S (if available)
    symbols_raw = None
    has_symbols = False
    try:
        if 'S' in ch_data.dtype.names:
            symbols_raw = ch_data['S']
            has_symbols = True
    except:
        try:
            symbols_raw = ch_data.get('S', None)
            has_symbols = symbols_raw is not None
        except:
            pass
    
    # Extract symbols if available - handle nested array structure from .mat files
    if has_symbols and symbols_raw is not None:
        try:
            if isinstance(symbols_raw, np.ndarray):
                # Handle nested array structure from scipy.io.loadmat
                while symbols_raw.ndim > 1 and symbols_raw.size > 0:
                    if symbols_raw.shape[0] == 1 or symbols_raw.shape[1] == 1:
                        symbols_raw = symbols_raw.flatten()
                    else:
                        break
                
                # Now convert to int, handling object arrays
                if symbols_raw.dtype == object:
                    # Extract from object array
                    symbols_list = []
                    for item in symbols_raw.flatten():
                        if isinstance(item, np.ndarray):
                            symbols_list.extend(item.flatten().tolist())
                        else:
                            symbols_list.append(item)
                    symbols = np.array(symbols_list, dtype=int)
                else:
                    symbols = symbols_raw.flatten().astype(int)
            else:
                symbols = np.array(symbols_raw).flatten().astype(int)
        except Exception as e:
            # If extraction fails, fall back to None
            symbols = None
    else:
        symbols = None
    
    # Get number of nonzero coefficients (fallback if no symbols)
    try:
        nonzero_raw = ch_data['nonzero_coeffs'] if 'nonzero_coeffs' in ch_data.dtype.names else None
    except:
        nonzero_raw = None
        
    if nonzero_raw is not None:
        if isinstance(nonzero_raw, np.ndarray):
            nonzero_coeffs = int(nonzero_raw.flatten()[0].item())
        else:
            nonzero_coeffs = int(nonzero_raw)
    else:
        nonzero_coeffs = 1024  # Default frame size
    
    # Total bits in Huffman bitstream
    total_bits = len(bitstream)
    
    # Estimate number of symbols (1024 for most frames, 128*8 for ESH)
    try:
        frame_type = frame_data['frame_type'] if 'frame_type' in frame_data.dtype.names else 'OLS'
    except:
        frame_type = 'OLS'
        
    if isinstance(frame_type, np.ndarray):
        frame_type = str(frame_type.flatten()[0])
    
    if 'ESH' in str(frame_type):
        num_symbols = 1024  # 128*8 for ESH
    else:
        num_symbols = 1024
    
    # Calculate tuples based on codebook
    tuple_size = CODEBOOK_TUPLE_SIZE.get(codebook, 2)
    num_tuples = num_symbols // tuple_size
    
    # Compute average codeword length per tuple
    L_tuple = total_bits / num_tuples if num_tuples > 0 else 0
    L_symbol = total_bits / num_symbols if num_symbols > 0 else 0
    
    # Compute EXACT entropy from quantized symbols
    if symbols is None:
        raise ValueError(f"Quantized symbols not available for frame {frame_idx}. "
                        "Please re-encode with updated aac_coder_3.py that saves symbols S.")
    
    # Extract tuples from quantized symbols
    tuples = extract_tuples(symbols, codebook)
    
    if len(tuples) == 0:
        raise ValueError(f"No tuples extracted from frame {frame_idx}")
    
    # Compute EXACT entropy from tuple distribution
    H_tuple, prob_dist = compute_entropy(tuples)
    H_symbol = H_tuple / tuple_size  # Per symbol
    
    # Efficiency metrics
    efficiency = H_tuple / L_tuple if L_tuple > 0 else 0
    redundancy = L_tuple - H_tuple
    shannon_satisfied = H_tuple <= L_tuple < H_tuple + 1 if H_tuple > 0 else True
    
    # Compression ratio
    uncompressed_bits = num_symbols * 16  # 16-bit fixed point
    compression_ratio = uncompressed_bits / total_bits if total_bits > 0 else 0
    
    if verbose:
        # Show most common tuples
        sorted_tuples = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 most common tuples:")
        for tup, prob in sorted_tuples:
            print(f"  {tup}: {prob*100:.2f}%")
    
    return {
        'frame_idx': frame_idx,
        'channel': channel,
        'codebook': codebook,
        'tuple_size': tuple_size,
        'num_tuples': len(tuples),
        'unique_tuples': len(prob_dist),
        'H_tuple': H_tuple,
        'H_symbol': H_symbol,
        'L_tuple': L_tuple,
        'L_symbol': L_symbol,
        'efficiency': efficiency,
        'redundancy': redundancy,
        'shannon_satisfied': shannon_satisfied,
        'compression_ratio': compression_ratio,
        'prob_dist': prob_dist,
        'total_bits': total_bits,
        'method': 'exact'
    }


def analyze_all_frames(aac_seq, max_frames=None, verbose=False):
    """
    Analyze entropy efficiency across all frames.
    
    Parameters
    ----------
    aac_seq : array
        AAC sequence from .mat file
    max_frames : int, optional
        Maximum frames to analyze (None = all)
    verbose : bool
        Print progress
    
    Returns
    -------
    results : list of dict
        Per-frame analysis results
    summary : dict
        Aggregate statistics
    """
    num_frames = len(aac_seq) if max_frames is None else min(max_frames, len(aac_seq))
    results = []
    
    for i in range(num_frames):
        result = analyze_frame_entropy(aac_seq[i], frame_idx=i, verbose=False)
        results.append(result)
    
    # Compute aggregate statistics
    H_tuples = [r['H_tuple'] for r in results]
    L_tuples = [r['L_tuple'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    redundancies = [r['redundancy'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    shannon_violations = sum(1 for r in results if not r['shannon_satisfied'])
    
    summary = {
        'num_frames': num_frames,
        'mean_H_tuple': np.mean(H_tuples),
        'std_H_tuple': np.std(H_tuples),
        'mean_L_tuple': np.mean(L_tuples),
        'std_L_tuple': np.std(L_tuples),
        'mean_efficiency': np.mean(efficiencies),
        'std_efficiency': np.std(efficiencies),
        'mean_redundancy': np.mean(redundancies),
        'std_redundancy': np.std(redundancies),
        'mean_compression_ratio': np.mean(compression_ratios),
        'shannon_violations': shannon_violations,
        'violation_rate': shannon_violations / num_frames * 100,
        'has_exact_entropy': True
    }
    
    return results, summary


def print_summary(results, summary):
    """Print formatted summary of entropy analysis."""
    print(f"\n{'='*70}")
    print("AAC HUFFMAN ENTROPY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nFrames: {summary['num_frames']} | Method: EXACT")
    print(f"\nEntropy (H):         {summary['mean_H_tuple']:.3f} bits/tuple")
    print(f"Codeword Length (L): {summary['mean_L_tuple']:.3f} bits/tuple")
    print(f"Efficiency (H/L):    {summary['mean_efficiency']*100:.1f}%")
    print(f"Redundancy (L-H):    {summary['mean_redundancy']:.3f} bits/tuple")
    
    print(f"\nCompression: {summary['mean_compression_ratio']:.2f}x (effective: {16/summary['mean_compression_ratio']:.2f} bits/symbol)")
    
    print(f"\n{'='*70}")


def visualize_entropy_analysis(results, summary, save_dir='level_3/outputs/plots/encoding_analysis'):
    """
    Create comprehensive visualization of entropy analysis as 4 separate plots.
    
    Parameters
    ----------
    results : list of dict
        Per-frame analysis results
    summary : dict
        Aggregate statistics
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== Plot 1: H vs L for sample frames ==========
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    sample_frames = results[:min(10, len(results))]
    frame_indices = [r['frame_idx'] for r in sample_frames]
    H_values = [r['H_tuple'] for r in sample_frames]
    L_values = [r['L_tuple'] for r in sample_frames]
    
    x = np.arange(len(frame_indices))
    width = 0.35
    
    ax1.bar(x - width/2, H_values, width, label='Entropy H', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, L_values, width, label='Avg Length L', color='#3498db', alpha=0.8)
    
    # Shannon's upper bound
    H_plus_1 = [h + 1 for h in H_values]
    ax1.plot(x, H_plus_1, 'r--', linewidth=2, label="Shannon's bound (H+1)", alpha=0.7)
    
    ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bits per Tuple', fontsize=11, fontweight='bold')
    ax1.set_title('EXACT Entropy vs Actual Codeword Length\n(Sample Frames)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frame_indices)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    fig1.tight_layout()
    save_path1 = os.path.join(save_dir, '1_entropy_vs_length.png')
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ========== Plot 4: H and L trend across all frames ==========
    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111)
    all_frame_indices = [r['frame_idx'] for r in results]
    all_H = [r['H_tuple'] for r in results]
    all_L = [r['L_tuple'] for r in results]
    
    ax4.plot(all_frame_indices, all_H, 'o-', color='#2ecc71', label='Entropy H', 
             markersize=3, linewidth=1, alpha=0.7)
    ax4.plot(all_frame_indices, all_L, 's-', color='#3498db', label='Avg Length L', 
             markersize=3, linewidth=1, alpha=0.7)
    
    # Mean lines
    ax4.axhline(summary['mean_H_tuple'], color='#27ae60', linestyle=':', linewidth=2, alpha=0.5)
    ax4.axhline(summary['mean_L_tuple'], color='#2980b9', linestyle=':', linewidth=2, alpha=0.5)
    
    ax4.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Bits per Tuple', fontsize=11, fontweight='bold')
    ax4.set_title(f'H and L Trend Across All Frames\n(n={len(results)})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    fig4.tight_layout()
    save_path4 = os.path.join(save_dir, '4_trend_across_frames.png')
    fig4.savefig(save_path4, dpi=300, bbox_inches='tight')
    plt.close(fig4)

def main():
    """Run entropy analysis on AAC encoded sequence."""
    
    # Load AAC encoded sequence
    aac_file = 'level_3/outputs/aac_seq_3.mat'
    
    try:
        mat_data = sio.loadmat(aac_file)
        
        # Handle different variable names
        if 'aac_seq_3' in mat_data:
            aac_seq = mat_data['aac_seq_3'].squeeze()
        else:
            print(f"Error: Could not find AAC sequence in file")
            return
        
        # Analyze all frames
        results, summary = analyze_all_frames(aac_seq, verbose=False)
        
        # Print summary
        print_summary(results, summary)
        
        # Visualize results
        visualize_entropy_analysis(results, summary)
        
        print(f"\n Analysis complete. Plots saved to: level_3/outputs/plots/encoding_analysis")
        
    except FileNotFoundError:
        print(f"\nError: File not found. Run demo_aac_3.py first.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()