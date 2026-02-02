# AAC Huffman Entropy Analysis Tool

## Overview

This tool analyzes the compression efficiency of your AAC encoder's Huffman coding by comparing the **average codeword length (L)** with the **Shannon entropy (H)**.

## Theory

**Shannon's Source Coding Theorem** states that for any lossless compression:
- **H(X)** = Entropy (theoretical minimum bits per symbol)
- **L** = Average codeword length (actual bits per symbol)
- **Theorem**: `H ≤ L < H + 1` for optimal Huffman coding

The closer L is to H, the more efficient the compression.

## Files

- **`utils_level_3/entropy_analysis.py`** - Core implementation (tuple-based analysis)
- **`demo_entropy.py`** - Main script to run analysis
- **`outputs/plots/encoding_analysis/entropy_analysis.png`** - Visualization output

## Usage

### Step 1: Encode Audio
First, generate an AAC encoded sequence:
```bash
python demo_aac_3.py
```

This creates `outputs/aac_seq_3.mat`

### Step 2: Run Entropy Analysis
```bash
python demo_entropy.py
```

## Output

### Console Output
- **Compression ratio**: How much smaller the encoded data is (e.g., 5.1x)
- **Effective bit rate**: Actual bits per symbol after encoding
- **Estimated efficiency**: How close to theoretical minimum (estimated)

### Visualization (`encoding_analysis/entropy_analysis.png`)
4-subplot figure showing:
1. **Top-Left**: Entropy H vs Codeword Length L for sample frames
2. **Top-Right**: Efficiency distribution across all frames  
3. **Bottom-Left**: Tuple probability distribution (most common tuples)
4. **Bottom-Right**: H and L trends across all frames

## Important Notes

### Current Limitation
⚠️ **Entropy values are ESTIMATED** because the quantized symbols (S) are not saved in the AAC sequence file. The estimates are based on coefficient sparsity (number of nonzero values).

### What We Can Accurately Measure
✅ **Compression ratio** - Uncompressed bits / Huffman bits  
✅ **Effective bit rate** - Actual bits per symbol  
✅ **Bitstream length** - Total Huffman encoded bits  
✅ **Average codeword length (L)** - Bits per tuple  

### What We Estimate
⚠️ **Entropy (H)** - Estimated from sparsity, not exact  

## For Exact Entropy Analysis

To get precise entropy vs codeword length comparison, modify the encoder:

1. Edit `aac_coder_3.py`
2. Save quantized symbols `S_chl` and `S_chr` in the output structure
3. Re-run encoding
4. Run this analysis again

The analysis tool will then compute exact tuple entropy from the actual symbol distribution.

## Interpretation

### Good Results
- **Compression ratio > 5x**: Excellent
- **Effective bit rate < 4 bits/symbol**: Good efficiency
- **High sparsity**: Many zeros = low entropy = better compression

### Example Output
```
Total Frames: 275
Average Compression Ratio: 5.05x
Effective Bit Rate: 3.17 bits/symbol
```

This means:
- Original data: 16 bits/symbol  
- After Huffman: 3.17 bits/symbol
- 80% size reduction achieved

## AAC-Specific: Tuple-Based Encoding

AAC uses **tuple-based** Huffman coding:
- **Codebooks 1-6**: 4-tuples (groups of 4 coefficients)
- **Codebooks 7-11**: 2-tuples (pairs of coefficients)

Shannon's bound applies to **tuple entropy**, not individual symbol entropy.

## Questions?

For theoretical background on Shannon entropy and Huffman coding:
- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- Huffman, D.A. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
