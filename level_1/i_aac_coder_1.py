import numpy as np
import soundfile as sf

from . import i_filter_bank

def i_aac_coder_1(aac_seq_1, filename_out):
    """
    Inverse AAC coder aac_coder_1() implementation.

    Args:
        aac_seq_1 (K length list)
        filename_out (str): output .wav file name where the decoded signal will be stored
                            Assumption: file includes 2 channel voice with fs = 48kHz
    Returns:
        x : decoded sample sequence
    """
    # Παράμετροι ανάλυσης
    N = 2048
    hop = N // 2
  
    num_frames = len(aac_seq_1)

    x = np.zeros((num_frames * hop + hop, 2))

    # Ανάκτηση κάθε πλαισίου και ανασύνθεση σήματος
    for i, frame in enumerate(aac_seq_1):

        frame_type = frame["frame_type"]
        win_type   = frame["win_type"]

        frame_T_L = i_filter_bank(frame["chl"]["frame_F"], frame_type, win_type)
        frame_T_R = i_filter_bank(frame["chr"]["frame_F"], frame_type, win_type)

        frame_T = np.stack([frame_T_L, frame_T_R], axis=1)

        start = i * hop
        x[start:start+N, :] += frame_T

    #Εγγραφή wav αρχείου 
    sf.write(filename_out, x, 48000)

    return x
