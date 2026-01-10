import numpy as np
import soundfile as sf

from i_filter_bank import i_filter_bank
from i_tns import i_tns

def i_aac_coder_2(aac_seq_2, filename_out):
    """
    Inverse AAC coder level 2 implementation (reverses aac_coder_2()).

    Args:
        aac_seq_2 (K length list): K is the number of frames that have been encoded.
                                   Each element of this list is a dictionary that includes:
                                   - aac_seq_2[i]["frame_type"] (str): Frame type of the i-th frame. 
                                                                        Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                   - aac_seq_2[i]["win_type"]: Window type
                                   - aac_seq_2[i]["chl"]["tns_coeffs"]: Quantized TNS coefficients for left channel
                                   - aac_seq_2[i]["chr"]["tns_coeffs"]: Quantized TNS coefficients for right channel
                                   - aac_seq_2[i]["chl"]["frame_F"]: MDCT coefficients for left channel after TNS
                                   - aac_seq_2[i]["chr"]["frame_F"]: MDCT coefficients for right channel after TNS
        filename_out (str): Output .wav file name where the decoded signal will be stored.
                           Assumption: File will include 2 channel audio with fs = 48kHz.
    
    Returns:
        x (array): Decoded sample sequence
    """
    # Analysis/synthesis parameters
    N = 2048
    hop = N // 2
  
    num_frames = len(aac_seq_2)
    x = np.zeros((num_frames * hop + hop, 2))

    # Process each frame
    for i, frame in enumerate(aac_seq_2):
        frame_type = frame["frame_type"]
        win_type = frame["win_type"]

        frame_F_L = i_tns(
            frame["chl"]["frame_F"],
            frame_type,
            frame["chl"]["tns_coeffs"]
        )

        frame_F_R = i_tns(
            frame["chr"]["frame_F"],
            frame_type,
            frame["chr"]["tns_coeffs"]
        )

        frame_T_L = i_filter_bank(frame_F_L, frame_type, win_type)
        frame_T_R = i_filter_bank(frame_F_R, frame_type, win_type)

        # Combine channels into a stereo frame
        frame_T = np.stack([frame_T_L, frame_T_R], axis=1)

        # Overlap-add reconstruction
        start = i * hop
        x[start:start+N, :] += frame_T

    # Save to output file
    sf.write(filename_out, x, 48000)

    return x
