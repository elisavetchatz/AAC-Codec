"""
Demo: Entropy Analysis for AAC Huffman Coding

This script demonstrates how to analyze the compression efficiency
of AAC's Huffman coding by comparing average codeword length with entropy.

Usage:
    python demo_entropy.py
"""

import scipy.io as sio
from entropy_analysis import (
    analyze_all_frames,
    print_summary,
    visualize_entropy_analysis
)


def main():
    """Run entropy analysis on AAC encoded sequence."""
    
    # Load AAC encoded sequence
    aac_file = '../../outputs/aac_seq_3.mat'
    
    try:
        print(f"\nLoading AAC sequence from: {aac_file}")
        mat_data = sio.loadmat(aac_file)
        
        # Handle different variable names
        if 'aac_seq_3' in mat_data:
            aac_seq = mat_data['aac_seq_3'].squeeze()
        elif 'AACSeq3' in mat_data:
            aac_seq = mat_data['AACSeq3'].squeeze()
        elif 'aac_seq_2' in mat_data:
            aac_seq = mat_data['aac_seq_2'].squeeze()
        elif 'AACSeq2' in mat_data:
            aac_seq = mat_data['AACSeq2'].squeeze()
        elif 'aac_seq_1' in mat_data:
            aac_seq = mat_data['aac_seq_1'].squeeze()
        elif 'AACSeq1' in mat_data:
            aac_seq = mat_data['AACSeq1'].squeeze()
        else:
            print(f"Error: Could not find AAC sequence in file")
            print(f"Available variables: {list(mat_data.keys())}")
            return
        
        print(f"Loaded {len(aac_seq)} frames\n")
        
        # Analyze all frames
        results, summary = analyze_all_frames(aac_seq, verbose=True)
        
        # Print summary
        print_summary(results, summary)
        
        # Visualize results
        visualize_entropy_analysis(results, summary)
        
        print("\n✓ Entropy analysis complete!")
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        print("\nLimitations:")
        print("- Entropy values are ESTIMATED (quantized symbols not stored)")
        print("- Estimates based on coefficient sparsity (nonzero count)")
        print("- For exact entropy analysis, modify encoder to save symbols")
        
        print("\nWhat we CAN conclude:")
        print(f"✓ Compression ratio: {summary['mean_compression_ratio']:.2f}x reduction")
        print(f"✓ Effective bit rate: {16/summary['mean_compression_ratio']:.2f} bits/symbol")
        print(f"✓ Huffman encoding achieves ~{100*(1-1/summary['mean_compression_ratio']):.1f}% size reduction")
        
        print("\nTo get EXACT entropy vs codeword length comparison:")
        print("1. Modify aac_coder_3.py to save quantized symbols S")
        print("2. Add S to the output structure for each channel")
        print("3. Re-run encoding and this analysis")
        print("="*70)
        
    except FileNotFoundError:
        print(f"\nError: File '{aac_file}' not found")
        print("Please run demo_aac_3.py first to generate the encoded sequence:")
        print("  python demo_aac_3.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
