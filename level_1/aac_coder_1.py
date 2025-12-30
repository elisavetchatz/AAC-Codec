import numpy as np
import soundfile as sf

from SSC import SSC
from filter_bank import filter_bank


def aac_coder_1(filename_in):
    """
    Args:
        filename_in (str): input .wav file name to be encoded
                            Assumption: file includes 2 channel voice with fs = 48kHz
    Returns:
        aac_seq_1 (K length list): K is the number of frames that have been encoded
                                    Each element of this list is a dictionary that includes:
                                    - aac_seq_1[i]['frame_type'] (str): frame type of the i-th frame. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
                                    - aac_seq_1[i]['win_type'] (str): window type ('KBD' or 'SIN')
                                    - aac_seq_1[i]['chl']["frame_F"]: MDCT coefficients of left channel 
                                          * (128, 8) for EIGHT SHORT SEQUENCE - each column is one subframe
                                          * (1024, 1) for other frame types
                                    - aac_seq_1[i]['chr']["frame_F"]: MDCT coefficients of right channel 
                                          * (128, 8) for EIGHT SHORT SEQUENCE - each column is one subframe
                                          * (1024, 1) for other frame types
    """
    x, fs = sf.read(filename_in)
    assert fs == 48000
    assert x.ndim == 2 and x.shape[1] == 2

    N = 2048
    hop = N // 2

    frames = []
    for start in range(0, x.shape[0] - N + 1, hop):
        frames.append(x[start:start + N, :])

    aac_seq_1 = []
    prev_frame_type = "OLS"

    for i in range(len(frames)):

        frame_T = frames[i]
        next_frame_T = frames[i + 1] if i + 1 < len(frames) else np.zeros_like(frame_T)

        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        win_type = "SIN"

        frame_F = filter_bank(frame_T, frame_type, win_type)

        if frame_type == "ESH":
            # Reshape to (128, 8) - each column is one subframe
            chl_F = frame_F[:, 0].reshape(128, 8, order='C')
            chr_F = frame_F[:, 1].reshape(128, 8, order='C')
        else:
            # Reshape to (1024, 1) - column vector
            chl_F = frame_F[:, 0].reshape(1024, 1)
            chr_F = frame_F[:, 1].reshape(1024, 1)

        aac_seq_1.append({
            "frame_type": frame_type,
            "win_type": win_type,
            "chl": {"frame_F": chl_F},
            "chr": {"frame_F": chr_F}
        })

        prev_frame_type = frame_type

    return aac_seq_1