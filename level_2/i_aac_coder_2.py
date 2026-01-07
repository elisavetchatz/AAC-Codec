from i_tns import i_tns
from level_1.i_aac_coder_1 import i_aac_coder_1

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
    aac_seq_1 = []
    # Remove TNS by applying inverse TNS on each frame
    for frame in aac_seq_2:
        frame_type = frame["frame_type"]
        win_type = frame["win_type"]

        # Inverse TNS per channel
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

        # Reconstruct compatible structure
        aac_seq_1.append({
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {"frame_F": frame_F_L},
            "chr": {"frame_F": frame_F_R}
        })

    # Delegate reconstruction to Level 1 inverse coder
    x = i_aac_coder_1(aac_seq_1, filename_out)
    
    return x
