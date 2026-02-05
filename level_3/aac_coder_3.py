import os
import sys
import numpy as np
import soundfile as sf
import scipy.io as sio

from filter_bank import filter_bank
from SSC import SSC
from tns import tns
from psycho import psycho
from aac_quantizer import aac_quantizer
from utils_level_3.huff_utils import load_LUT, encode_huff


def aac_coder_3(filename_in, filename_aac_coded):
    """
    AAC coder level 3 implementation with psychoacoustic model and Huffman coding.

    Args:
        filename_in (str): Input .wav file name to be encoded.
                          Assumption: File includes 2 channel audio with fs = 48kHz.
        filename_aac_coded (str): Output .mat file where the aac_seq_3 structure will be saved.
    
    Returns:
        aac_seq_3 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_3[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_3[i]["win_type"]: Window type
                                   - aac_seq_3[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_3[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_3[i]["chl"]["T"]: Psychoacoustic model thresholds (NBx1) for left channel
                                                               (for visualization purposes, not needed for encoding)
                                   - aac_seq_3[i]["chr"]["T"]: Psychoacoustic model thresholds (NBBx1) for right channel
                                                               (for visualization purposes, not needed for encoding)
                                   - aac_seq_3[i]["chl"]["G"]: Quantized global gains (1 or 8) for left channel
                                   - aac_seq_3[i]["chr"]["G"]: Quantized global gains (1 or 8) for right channel
                                   - aac_seq_3[i]["chl"]["sfc"]: Huffman encoded sequence of sfc for left channel
                                   - aac_seq_3[i]["chr"]["sfc"]: Huffman encoded sequence of sfc for right channel
                                   - aac_seq_3[i]["chl"]["stream"]: Huffman encoded sequence of quantized MDCT 
                                                                    coefficients for left channel
                                   - aac_seq_3[i]["chr"]["stream"]: Huffman encoded sequence of quantized MDCT 
                                                                    coefficients for right channel
                                   - aac_seq_3[i]["chl"]["codebook"]: Huffman codebook used for left channel
                                   - aac_seq_3[i]["chr"]["codebook"]: Huffman codebook used for right channel
    """
    
    x, fs = sf.read(filename_in)
    if fs != 48000 or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Input must be 48kHz stereo audio")

    N = 2048
    hop = N // 2
    frames = []
    for start in range(0, x.shape[0] - N + 1, hop):
        frames.append(x[start:start + N, :])

    prev_frame_type = "OLS"
    aac_seq_3 = []

    huffLUT = load_LUT()

    # Process each frame
    for i in range(len(frames)):
        print(f"Encoding frame {i + 1} of {len(frames)}")

        frame_T = frames[i]
        next_frame_T = frames[i + 1] if i + 1 < len(frames) else np.zeros_like(frame_T)
        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        
        win_type = "SIN"
        frame_F = filter_bank(frame_T, frame_type, win_type)

        if frame_type == "ESH":
            chl_F = frame_F[:, :, 0]
            chr_F = frame_F[:, :, 1]
        else:
            chl_F = frame_F[:, 0]
            chr_F = frame_F[:, 1]

        chl_F_tns, tns_chl = tns(chl_F, frame_type)
        chr_F_tns, tns_chr = tns(chr_F, frame_type)

        # psychoacoustic model
        frame_T_prev_1 = frames[i - 1] if i >= 1 else np.zeros_like(frame_T)
        frame_T_prev_2 = frames[i - 2] if i >= 2 else np.zeros_like(frame_T)

        SMR_chl = psycho(frame_T[:, 0], frame_type, frame_T_prev_1[:, 0], frame_T_prev_2[:, 0])
        SMR_chr = psycho(frame_T[:, 1], frame_type, frame_T_prev_1[:, 1], frame_T_prev_2[:, 1])

        # Quantization
        S_chl, sfc_chl, G_chl = aac_quantizer(chl_F_tns, frame_type, SMR_chl)
        S_chr, sfc_chr, G_chr = aac_quantizer(chr_F_tns, frame_type, SMR_chr)

        # Calculate sparsity metrics
        nonzero_chl = int(np.count_nonzero(S_chl))
        nonzero_chr = int(np.count_nonzero(S_chr))
        total_coeffs_frame = int(S_chl.size)
        
        # Coefficient magnitude statistics
        max_coeff_chl = int(np.max(np.abs(S_chl)))
        max_coeff_chr = int(np.max(np.abs(S_chr)))
        
        # Print first frame statistics
        if i == 0:
            print(f"\nFrame {i} - Coefficient Statistics:")
            print(f"  CHL - Non-zero: {nonzero_chl}/{total_coeffs_frame} ({100*nonzero_chl/total_coeffs_frame:.1f}%), Max value: {max_coeff_chl}")
            print(f"  CHR - Non-zero: {nonzero_chr}/{total_coeffs_frame} ({100*nonzero_chr/total_coeffs_frame:.1f}%), Max value: {max_coeff_chr}")
        
        # Huffman encoding
        stream_chl, codebook_chl = encode_huff(S_chl.flatten().astype(int), huffLUT)
        stream_chr, codebook_chr = encode_huff(S_chr.flatten().astype(int), huffLUT)
        sfc_stream_chl, sfc_codebook_chl = encode_huff(sfc_chl.flatten().astype(int), huffLUT)
        sfc_stream_chr, sfc_codebook_chr = encode_huff(sfc_chr.flatten().astype(int), huffLUT)

        aac_seq_3.append({
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {
                "tns_coeffs": tns_chl,
                "T": SMR_chl,  # For visualization
                "G": G_chl,
                "sfc": sfc_stream_chl,
                "sfc_codebook": sfc_codebook_chl,
                "stream": stream_chl,
                "codebook": codebook_chl,
                "S": S_chl.flatten(),  # Quantized symbols for entropy analysis
                "nonzero_coeffs": nonzero_chl
            },
            "chr": {
                "tns_coeffs": tns_chr,
                "T": SMR_chr,  # For visualization
                "G": G_chr,
                "sfc": sfc_stream_chr,
                "sfc_codebook": sfc_codebook_chr,
                "stream": stream_chr,
                "codebook": codebook_chr,
                "S": S_chr.flatten(),  # Quantized symbols for entropy analysis
                "nonzero_coeffs": nonzero_chr
            },
            "total_coeffs": total_coeffs_frame
        })

        # Update previous frame type
        prev_frame_type = frame_type

    # Save encoded sequence to .mat file
    sio.savemat(filename_aac_coded, {'aac_seq_3': aac_seq_3})
    print(f"Encoded sequence saved to {filename_aac_coded}")

    return aac_seq_3
