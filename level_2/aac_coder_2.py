import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_2.tns import tns
from level_1.aac_coder_1 import aac_coder_1

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
        frames (list): Time-domain frames (K length list of 2048x2 arrays)
    """
    # Get Level 1 encoded sequence (SSC + Filterbank) and time-domain frames
    aac_seq_1, frames = aac_coder_1(filename_in)
    aac_seq_2 = []

    # Process each frame independently
    for frame in aac_seq_1:

        frame_type = frame["frame_type"]
        win_type = frame["win_type"]

        # Apply TNS independently to left and right channels
        chl_out, tns_chl = tns(frame["chl"]["frame_F"], frame_type)
        chr_out, tns_chr = tns(frame["chr"]["frame_F"], frame_type)

        # Store Level 2 encoded frame
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

    return aac_seq_2, frames
