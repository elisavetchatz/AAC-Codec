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
    aac_file = 'C:\\Users\\30690\\AAC-Codec\\level_3\\outputs\\aac_seq_3.mat'
    
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
        
        if summary['has_exact_entropy']:
            print("\n✅ Using EXACT entropy computation from quantized symbols")
            print("\nKey Findings:")
            print(f"✓ Shannon Entropy (H): {summary['mean_H_tuple']:.3f} bits/tuple")
            print(f"✓ Huffman Length (L):  {summary['mean_L_tuple']:.3f} bits/tuple")
            print(f"✓ Coding Efficiency:   {summary['mean_efficiency']*100:.2f}% (H/L)")
            print(f"✓ Redundancy:          {summary['mean_redundancy']:.3f} bits/tuple")
            
            print(f"\n✓ Compression ratio: {summary['mean_compression_ratio']:.2f}x reduction")
            print(f"✓ Effective bit rate: {16/summary['mean_compression_ratio']:.2f} bits/symbol")
            print(f"✓ Huffman encoding achieves ~{100*(1-1/summary['mean_compression_ratio']):.1f}% size reduction")
            
            print("\nWhat the efficiency tells us:")
            if summary['mean_efficiency'] > 0.85:
                print("✅ EXCELLENT: Huffman coding is near-optimal for this data")
            elif summary['mean_efficiency'] > 0.65:
                print("✓ GOOD: Huffman coding is reasonably efficient")
            elif summary['mean_efficiency'] > 0.45:
                print("⚠ MODERATE: Some redundancy exists - room for improvement")
            else:
                print("⚠ LOW: Significant redundancy - consider better codebook selection")
            
            print(f"\nRedundancy analysis:")
            print(f"  Average waste: {summary['mean_redundancy']:.2f} bits/tuple")
            print(f"  This represents {summary['mean_redundancy']/summary['mean_L_tuple']*100:.1f}% of encoded size")
            
        else:
            print("\n⚠ Using ESTIMATED entropy (quantized symbols not available)")
            print("\nLimitations:")
            print("- Entropy values are estimated from coefficient sparsity")
            print("- For exact analysis, re-encode with updated aac_coder_3.py")
            
            print("\nWhat we CAN conclude:")
            print(f"✓ Compression ratio: {summary['mean_compression_ratio']:.2f}x reduction")
            print(f"✓ Effective bit rate: {16/summary['mean_compression_ratio']:.2f} bits/symbol")
            print(f"✓ Huffman encoding achieves ~{100*(1-1/summary['mean_compression_ratio']):.1f}% size reduction")
        
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
