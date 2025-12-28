def SSC(frame_T, next_frame_T, prev_frame_type):
    """
    Sequence Segmentation Control implementation.

    Args:
        frame_T (2048x2 array): frame i in time domain. Consists of 2 voice channels. 
        next_frame_T (2048x2 array): frame i+1 in time domain. We use it to select window
        prev_frame_type (str): frame type of i-1 frame. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
    
    Returns:
        frame_type (str): frame type of current frame. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
    """

    return frame_type