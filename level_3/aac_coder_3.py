import os
import sys
import numpy as np
import soundfile as sf
import scipy.io as sio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_1.aac_coder_1 import aac_coder_1
from level_2.aac_coder_2 import aac_coder_2
from level_3.psycho import psycho
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
    
    aac_seq_2, frames = aac_coder_2(filename_in)

    huffLUT = load_LUT()
    
    aac_seq_3 = []

    # Process each frame independently
    for i, frame in enumerate(aac_seq_2):
        print(f"Level 3 encoding frame {i + 1} of {len(aac_seq_2)}")
        
        frame_type = frame["frame_type"]
        win_type = frame["win_type"]

        # Get time-domain frames for psychoacoustic model
        frame_T = frames[i]
        frame_T_prev_1 = frames[i - 1] if i >= 1 else np.zeros_like(frame_T)
        frame_T_prev_2 = frames[i - 2] if i >= 2 else np.zeros_like(frame_T)

        SMR_chl = psycho(frame_T[:, 0], frame_type, frame_T_prev_1[:, 0], frame_T_prev_2[:, 0])
        SMR_chr = psycho(frame_T[:, 1], frame_type, frame_T_prev_1[:, 1], frame_T_prev_2[:, 1])

        S_chl, sfc_chl, G_chl = aac_quantizer(frame["chl"]["frame_F"], frame_type, SMR_chl)
        S_chr, sfc_chr, G_chr = aac_quantizer(frame["chr"]["frame_F"], frame_type, SMR_chr)

        stream_chl, codebook_chl = encode_huff(S_chl.flatten().astype(int), huffLUT)
        stream_chr, codebook_chr = encode_huff(S_chr.flatten().astype(int), huffLUT)
        sfc_stream_chl, sfc_codebook_chl = encode_huff(sfc_chl.flatten().astype(int), huffLUT)
        sfc_stream_chr, sfc_codebook_chr = encode_huff(sfc_chr.flatten().astype(int), huffLUT)

        aac_seq_3.append({
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {
                "tns_coeffs": frame["chl"]["tns_coeffs"],
                "T": SMR_chl,  # For visualization
                "G": G_chl,
                "sfc": sfc_stream_chl,
                "sfc_codebook": sfc_codebook_chl,
                "stream": stream_chl,
                "codebook": codebook_chl
            },
            "chr": {
                "tns_coeffs": frame["chr"]["tns_coeffs"],
                "T": SMR_chr,  # For visualization
                "G": G_chr,
                "sfc": sfc_stream_chr,
                "sfc_codebook": sfc_codebook_chr,
                "stream": stream_chr,
                "codebook": codebook_chr
            }
        })

    # Save encoded sequence to .mat file
    sio.savemat(filename_aac_coded, {'aac_seq_3': aac_seq_3})
    print(f"Encoded sequence saved to {filename_aac_coded}")

    return aac_seq_3
