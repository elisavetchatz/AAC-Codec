import os
import sys
import numpy as np
import soundfile as sf

from utils_level_3.huff_utils import load_LUT, decode_huff
from i_aac_quantizer import i_aac_quantizer
from i_tns import i_tns
from i_filter_bank import i_filter_bank


def i_aac_coder_3(aac_seq_3, filename_out):
    """
    Inverse AAC coder level 3 implementation (reverses aac_coder_3()).

    Args:
        aac_seq_3 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_3[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_3[i]["win_type"]: Window type
                                   - aac_seq_3[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_3[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_3[i]["chl"]["T"]: Psychoacoustic model thresholds (NB×1) for left channel
                                   - aac_seq_3[i]["chr"]["T"]: Psychoacoustic model thresholds (NB×1) for right channel
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
        filename_out (str): Output .wav file name where the decoded signal will be stored.
                           Assumption: File will include 2 channel audio with fs = 48kHz.
    
    Returns:
        x (array): Decoded sample sequence
    """

    huffLUT = load_LUT()
    decoded_frames = []

    for i, frame in enumerate(aac_seq_3):
        print(f"Level 3 decoding frame {i + 1} of {len(aac_seq_3)}")
        
        frame_type = frame["frame_type"]
        win_type = frame["win_type"]

        codebook_chl = frame["chl"]["codebook"]
        if codebook_chl == 0:
            S_chl = np.zeros(1024, dtype=int)
        else:
            S_chl = np.array(decode_huff(frame["chl"]["stream"], huffLUT[codebook_chl]))

        codebook_chr = frame["chr"]["codebook"]
        if codebook_chr == 0:
            S_chr = np.zeros(1024, dtype=int)
        else:
            S_chr = np.array(decode_huff(frame["chr"]["stream"], huffLUT[codebook_chr]))

        sfc_chl = frame["chl"]["sfc"]

        sfc_chr = frame["chr"]["sfc"]

        frame_F_chl = i_aac_quantizer(S_chl, sfc_chl, frame["chl"]["G"], frame_type)
        frame_F_chr = i_aac_quantizer(S_chr, sfc_chr, frame["chr"]["G"], frame_type)
        frame_F_chl = i_tns(frame_F_chl, frame_type, frame["chl"]["tns_coeffs"])
        frame_F_chr = i_tns(frame_F_chr, frame_type, frame["chr"]["tns_coeffs"])

        # Apply inverse filterbank to each channel separately
        frame_T_chl = i_filter_bank(frame_F_chl, frame_type, win_type)
        frame_T_chr = i_filter_bank(frame_F_chr, frame_type, win_type)
        
        # Combine channels (2048, 2)
        frame_T = np.column_stack([frame_T_chl, frame_T_chr])
        
        decoded_frames.append(frame_T)

    # Overlap-add reconstruction
    N = 2048
    hop = N // 2
    total_samples = (len(decoded_frames) - 1) * hop + N
    x = np.zeros((total_samples, 2))

    for i, frame in enumerate(decoded_frames):
        start = i * hop
        end = start + N
        x[start:end, :] += frame

    # Save to WAV file
    sf.write(filename_out, x, 48000)
    print(f"Decoded audio saved to {filename_out}")

    return x
