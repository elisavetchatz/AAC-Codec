import os
import sys
import numpy as np
import soundfile as sf

from filter_bank import filter_bank
from SSC import SSC
from tns import tns

def aac_coder_2(filename_in):
    """
    AAC coder level 2 implementation with TNS support.

    Args:
        filename_in (str): Input .wav file name to be encoded.
                          Assumption: File includes 2 channel audio with fs = 48kHz.
    
    Returns:
        aac_seq_2 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_2[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_2[i]["win_type"]: Window type
                                   - aac_seq_2[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_2[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_2[i]["chl"]["frame_F"]: MDCT coefficients for left channel after TNS
                                   - aac_seq_2[i]["chr"]["frame_F"]: MDCT coefficients for right channel after TNS
    """
    # Load audio file
    x, fs = sf.read(filename_in)
    if fs != 48000 or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Input must be 48kHz stereo audio")

    N = 2048
    hop = N // 2

    # Split signal into overlapping frames (50% overlap)
    frames = []
    for start in range(0, x.shape[0] - N + 1, hop):
        frames.append(x[start:start + N, :])

    aac_seq_2 = []
    prev_frame_type = "OLS"

    # Process each frame
    for i in range(len(frames)):
        print(f"Encoding frame {i + 1} of {len(frames)}")

        # Current frame in time domain
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

        chl_out, tns_chl = tns(chl_F, frame_type)
        chr_out, tns_chr = tns(chr_F, frame_type)

        aac_seq_2.append({
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {
                "frame_F": chl_out,
                "tns_coeffs": tns_chl
            },
            "chr": {
                "frame_F": chr_out,
                "tns_coeffs": tns_chr
            }
        })

        # Update previous frame type
        prev_frame_type = frame_type

    return aac_seq_2
