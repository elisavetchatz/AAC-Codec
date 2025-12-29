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
    from scipy.signal import lfilter
    import numpy as np
    
    b = [0.7548, -0.7548]
    a = [1.0, -0.5095]
    s2 = np.zeros((2, 8))
    ds2 = np.zeros((2, 8))
    is_esh_next = [False, False]

    for ch in range(2):
        # Υψιπερατό φίλτρο
        y = lfilter(b, a, next_frame_T[:, ch])

        # Υπολογισμός s_l^2
        for l in range(8):
            block = y[l*128:(l+1)*128]
            s2[ch, l] = np.sum(block ** 2)
        # Υπολογισμός ds_l^2
        for l in range(1, 8):
            prev_mean = np.mean(s2[ch, :l])
            # Έλεγχος ESH
            if prev_mean > 0:
                ds2[ch, l] = s2[ch, l] / prev_mean
            if s2[ch, l] > 1e-3 and ds2[ch, l] > 10:
                is_esh_next[ch] = True
                break
    
    next_is_esh = is_esh_next[0] or is_esh_next[1]

    # Τελική απόφαση frame_type 
    if prev_frame_type == "OLS":
        frame_type = "LSS" if next_is_esh else "OLS"

    elif prev_frame_type == "ESH":
        frame_type = "ESH" if next_is_esh else "LPS"

    elif prev_frame_type == "LSS":
        frame_type = "ESH"

    elif prev_frame_type == "LPS":
        frame_type = "OLS"

    else:
        raise ValueError("Invalid previous frame type")


    return frame_type