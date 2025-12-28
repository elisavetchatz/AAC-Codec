def filter_bank(frame_T, frame_type, win_type):
    """
    Filter Bank implementation.

    Args:
        frame_T (2048x2 array): frame in time domain
        frame_type (str): frame type of the current frame
        win_type (str): type of the weight window of the current frame. Can be 'KBD' or 'SIN'

    Returns:
        frame_F (1024x2 array): frame in frequency domain in an MDCT coefficient format
                                Consists of either:
                                a) the coefficients of the 2 channels for 'OLS', 'LSS', 'LPS' frame types or
                                b) 8 (128x2) subarrays for each subframe for 'ESH' frame type, placed in rows according to subframe order
    """

    return frame_F