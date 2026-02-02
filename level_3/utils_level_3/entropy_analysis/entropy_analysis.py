"""
Entropy Analysis for AAC Huffman Coding

This module analyzes the compression efficiency of AAC's tuple-based Huffman coding
by comparing the average codeword length (L) with the Shannon entropy (H).

Shannon's Theorem: H ≤ L < H + 1 (for optimal Huffman coding)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os


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
        codebook = int(np.median(codebook_arr[codebook_arr > 0])) if len(codebook_arr[codebook_arr > 0]) > 0 else 3
    else:
        codebook = int(codebook_raw)
    
    # Get number of nonzero coefficients to estimate total symbols
    try:
        nonzero_raw = ch_data['nonzero_coeffs'] if 'nonzero_coeffs' in ch_data.dtype.names else None
    except:
        nonzero_raw = None
        
    if nonzero_raw is not None:
        if isinstance(nonzero_raw, np.ndarray):
            nonzero_coeffs = int(nonzero_raw.flatten()[0])
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
    
    # Since we don't have the original quantized symbols, we estimate entropy
    # based on the sparsity (nonzero coefficients)
    # For a sparse distribution with many zeros:
    # H ≈ -p_zero * log2(p_zero) - p_nonzero * log2(p_nonzero/num_unique_nonzero)
    p_zero = 1 - (nonzero_coeffs / num_symbols)
    p_nonzero = nonzero_coeffs / num_symbols
    
    # Estimate entropy (conservative estimate)
    # Assume nonzero values are distributed among ~10 unique values (typical for AAC)
    if p_zero > 0 and p_zero < 1:
        H_symbol_est = -p_zero * np.log2(p_zero) if p_zero > 0 else 0
        if p_nonzero > 0:
            # Assume uniform distribution among nonzero values
            H_symbol_est += p_nonzero * np.log2(10)  # ~10 typical nonzero values
    else:
        H_symbol_est = 0.0
    
    H_tuple = H_symbol_est * tuple_size
    
    # Efficiency metrics
    efficiency = (H_tuple / L_tuple * 100) if L_tuple > 0 and H_tuple > 0 else 100
    redundancy = L_tuple - H_tuple
    shannon_satisfied = H_tuple <= L_tuple < H_tuple + 1 if H_tuple > 0 else True
    
    # Compression ratio
    uncompressed_bits = num_symbols * 16  # 16-bit fixed point
    compression_ratio = uncompressed_bits / total_bits if total_bits > 0 else 0
    
    results = {
        'frame_idx': frame_idx,
        'codebook': codebook,
        'num_symbols': num_symbols,
        'num_tuples': num_tuples,
        'entropy_tuple': H_tuple,
        'avg_length_tuple': L_tuple,
        'entropy_symbol': H_symbol_est,
        'avg_length_symbol': L_symbol,
        'efficiency': efficiency,
        'redundancy': redundancy,
        'shannon_satisfied': shannon_satisfied,
        'compression_ratio': compression_ratio,
        'nonzero_coeffs': nonzero_coeffs,
        'sparsity': p_zero * 100,  # percentage of zeros
        'tuple_probs': {}  # Empty since we don't have actual symbols
    }
    
    if verbose:
        print(f"\n=== Frame {frame_idx} Analysis ({channel} channel) ===")
        print(f"Codebook: {codebook} ({tuple_size}-tuple encoding)")
        print(f"Symbols: {num_symbols}, Tuples: {num_tuples}")
        print(f"Nonzero coeffs: {nonzero_coeffs} ({100-p_zero*100:.1f}%)")
        print(f"\nEstimated Tuple Entropy (H):  {H_tuple:.4f} bits/tuple")
        print(f"Avg Codeword Length (L):       {L_tuple:.4f} bits/tuple")
        print(f"Efficiency:                    {efficiency:.2f}%")
        print(f"Redundancy:                    {redundancy:.4f} bits/tuple")
        print(f"Shannon's bound: {'✓' if shannon_satisfied else '✗'} ({H_tuple:.2f} ≤ {L_tuple:.2f} < {H_tuple+1:.2f})")
        print(f"\nPer-symbol: H≈{H_symbol_est:.4f}, L={L_symbol:.4f} bits/symbol")
        print(f"Compression ratio: {compression_ratio:.2f}x vs 16-bit")
        print(f"Total bitstream: {total_bits} bits")
    
    return results


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
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing {num_frames} AAC frames...")
        print(f"{'='*70}")
    
    for i in range(num_frames):
        if verbose and i % 10 == 0:
            print(f"Processing frame {i}/{num_frames}...")
        
        result = analyze_frame_entropy(aac_seq[i], frame_idx=i, verbose=False)
        results.append(result)
    
    # Compute aggregate statistics
    entropies_tuple = [r['entropy_tuple'] for r in results]
    lengths_tuple = [r['avg_length_tuple'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    redundancies = [r['redundancy'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    shannon_violations = sum(1 for r in results if not r['shannon_satisfied'])
    
    summary = {
        'num_frames': num_frames,
        'mean_entropy_tuple': np.mean(entropies_tuple),
        'std_entropy_tuple': np.std(entropies_tuple),
        'mean_avg_length_tuple': np.mean(lengths_tuple),
        'std_avg_length_tuple': np.std(lengths_tuple),
        'mean_efficiency': np.mean(efficiencies),
        'std_efficiency': np.std(efficiencies),
        'mean_redundancy': np.mean(redundancies),
        'std_redundancy': np.std(redundancies),
        'mean_compression_ratio': np.mean(compression_ratios),
        'shannon_violations': shannon_violations,
        'violation_rate': shannon_violations / num_frames * 100
    }
    
    return results, summary


def print_summary(results, summary):
    """Print formatted summary of entropy analysis."""
    print(f"\n{'='*70}")
    print("AAC HUFFMAN ENTROPY ANALYSIS - SUMMARY")
    print(f"{'='*70}")
    print(f"Total Frames Analyzed: {summary['num_frames']}")
    print(f"\nNOTE: Entropy values are ESTIMATED from coefficient sparsity")
    print(f"      (actual quantized symbols not stored in AAC sequence)")
    print(f"      For exact entropy, symbols would need to be saved during encoding.")
    print(f"\n--- AGGREGATE RESULTS (Tuple-Based) ---")
    print(f"Estimated Entropy (H):       {summary['mean_entropy_tuple']:.4f} ± {summary['std_entropy_tuple']:.4f} bits/tuple")
    print(f"Average Codeword Length (L): {summary['mean_avg_length_tuple']:.4f} ± {summary['std_avg_length_tuple']:.4f} bits/tuple")
    print(f"Estimated Efficiency (H/L):  {summary['mean_efficiency']:.2f} ± {summary['std_efficiency']:.2f}%")
    print(f"Redundancy (L-H):            {summary['mean_redundancy']:.4f} ± {summary['std_redundancy']:.4f} bits/tuple")
    print(f"\nOverall Compression Ratio:   {summary['mean_compression_ratio']:.2f}x (vs 16-bit fixed)")
    print(f"\n--- COMPRESSION INTERPRETATION ---")
    if summary['mean_compression_ratio'] >= 5:
        print(f"✓ Excellent compression achieved ({summary['mean_compression_ratio']:.1f}x)")
    else:
        print(f"○ Moderate compression ({summary['mean_compression_ratio']:.1f}x)")
    print(f"  Effective bit rate: {16/summary['mean_compression_ratio']:.2f} bits/symbol")
    
    # Show first few frames
    print(f"\n--- SAMPLE FRAME DETAILS (First 5 Frames) ---")
    print(f"{'Frame':<8} {'CB':<4} {'H(est)':<12} {'L(actual)':<12} {'Eff%':<10} {'Comp.Ratio':<12}")
    print("-" * 70)
    for r in results[:5]:
        print(f"{r['frame_idx']:<8} {r['codebook']:<4} {r['entropy_tuple']:<12.4f} {r['avg_length_tuple']:<12.4f} {r['efficiency']:<10.2f} {r['compression_ratio']:<12.2f}x")
    
    if len(results) > 5:
        print("...")
    
    print(f"{'='*70}\n")


def visualize_entropy_analysis(results, summary, save_dir='outputs/plots/encoding_analysis'):
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
    H_values = [r['entropy_tuple'] for r in sample_frames]
    L_values = [r['avg_length_tuple'] for r in sample_frames]
    
    x = np.arange(len(frame_indices))
    width = 0.35
    
    ax1.bar(x - width/2, H_values, width, label='Entropy H', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, L_values, width, label='Avg Length L', color='#3498db', alpha=0.8)
    
    # Shannon's upper bound
    H_plus_1 = [h + 1 for h in H_values]
    ax1.plot(x, H_plus_1, 'r--', linewidth=2, label="Shannon's bound (H+1)", alpha=0.7)
    
    ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bits per Tuple', fontsize=11, fontweight='bold')
    ax1.set_title('Estimated Entropy vs Actual Codeword Length\n(Sample Frames)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frame_indices)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.02, 0.98, 'Note: Entropy estimated from sparsity', 
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig1.tight_layout()
    save_path1 = os.path.join(save_dir, '1_entropy_vs_length.png')
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot 1 saved: {save_path1}")
    plt.close(fig1)
    
    # ========== Plot 2: Efficiency distribution ==========
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    efficiencies = [r['efficiency'] for r in results]
    ax2.hist(efficiencies, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.axvline(summary['mean_efficiency'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {summary['mean_efficiency']:.2f}%")
    ax2.set_xlabel('Efficiency (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Frames', fontsize=11, fontweight='bold')
    ax2.set_title(f'Efficiency Distribution Across All Frames\n(n={len(results)})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    fig2.tight_layout()
    save_path2 = os.path.join(save_dir, '2_efficiency_distribution.png')
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 2 saved: {save_path2}")
    plt.close(fig2)
    
    # ========== Plot 3: Tuple probability distribution ==========
    fig3 = plt.figure(figsize=(12, 6))
    ax3 = fig3.add_subplot(111)
    repr_frame = results[0]  # First frame as representative
    tuple_probs = repr_frame.get('tuple_probs', {})
    
    if tuple_probs:
        # Get top 20 most frequent tuples
        sorted_tuples = sorted(tuple_probs.items(), key=lambda x: x[1], reverse=True)[:20]
        tuple_labels = [str(t) for t, p in sorted_tuples]
        tuple_probs_values = [p for t, p in sorted_tuples]
        
        ax3.bar(range(len(tuple_labels)), tuple_probs_values, color='#e74c3c', alpha=0.7)
        ax3.set_xlabel('Tuple', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax3.set_title(f'Tuple Probability Distribution\n(Frame {repr_frame["frame_idx"]}, Top 20)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(tuple_labels)))
        ax3.set_xticklabels(tuple_labels, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
    else:
        # Show sparsity info instead
        sparsity_data = [r['sparsity'] for r in results[:20]]
        frame_idx_data = [r['frame_idx'] for r in results[:20]]
        ax3.bar(range(len(sparsity_data)), sparsity_data, color='#e74c3c', alpha=0.7)
        ax3.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Sparsity (%)', fontsize=11, fontweight='bold')
        ax3.set_title(f'Coefficient Sparsity (First 20 Frames)\n(Higher = More Zeros = Lower Entropy)', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(frame_idx_data)))
        ax3.set_xticklabels(frame_idx_data)
        ax3.grid(axis='y', alpha=0.3)
    
    fig3.tight_layout()
    save_path3 = os.path.join(save_dir, '3_probability_or_sparsity.png')
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 3 saved: {save_path3}")
    plt.close(fig3)
    
    # ========== Plot 4: H and L trend across all frames ==========
    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111)
    ax4 = fig4.add_subplot(111)
    all_frame_indices = [r['frame_idx'] for r in results]
    all_H = [r['entropy_tuple'] for r in results]
    all_L = [r['avg_length_tuple'] for r in results]
    
    ax4.plot(all_frame_indices, all_H, 'o-', color='#2ecc71', label='Entropy H', 
             markersize=3, linewidth=1, alpha=0.7)
    ax4.plot(all_frame_indices, all_L, 's-', color='#3498db', label='Avg Length L', 
             markersize=3, linewidth=1, alpha=0.7)
    
    # Mean lines
    ax4.axhline(summary['mean_entropy_tuple'], color='#27ae60', linestyle=':', linewidth=2, alpha=0.5)
    ax4.axhline(summary['mean_avg_length_tuple'], color='#2980b9', linestyle=':', linewidth=2, alpha=0.5)
    
    ax4.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Bits per Tuple', fontsize=11, fontweight='bold')
    ax4.set_title(f'H and L Trend Across All Frames\n(n={len(results)})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    fig4.tight_layout()
    save_path4 = os.path.join(save_dir, '4_trend_across_frames.png')
    fig4.savefig(save_path4, dpi=300, bbox_inches='tight')
    print(f"✓ Plot 4 saved: {save_path4}")
    plt.close(fig4)
    
    print(f"\n✓ All 4 plots saved to: {save_dir}")

