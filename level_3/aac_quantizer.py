import numpy as np

from quantizer_utils import (
    load_scalefactor_bands,
    compute_thresholds,
    initial_alpha_estimate,
    build_alpha_per_coeff,
    quantize,
    dequantize, 
    band_error_power
)
MAX_SFC_DIFF = 60
def aac_quantizer(frame_F, frame_type, SMR):
    """
    Quantizer stage implementation for one channel.
    Internally calculates the hearing threshold T(b) and implements quantization.

    Args:
        frame_F (array): MDCT coefficients
                        Dimensions: 128×8 for EIGHT_SHORT_SEQUENCE, 1024×1 otherwise
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
        SMR (array): Signal to Mask Ratio
                    Dimensions: 42×8 for EIGHT_SHORT_SEQUENCE, 69×1 otherwise
    
    Returns:
        S (array): Array of quantized MDCT coefficient symbols for current frame
                  Dimensions: 1024×1 (for all frame types)
        sfc (array): Scalefactor values for each Scalefactor band
                    Dimensions: NB×8 for EIGHT_SHORT_SEQUENCE frames (where NB is number of bands),
                               NB×1 for all other types
        G (array or float): Global gain of current frame
                           Dimensions: 1×8 for EIGHT_SHORT_SEQUENCE or single value for all other cases
    """
    # Frame handling
    if frame_type == "ESH":
        X = np.asarray(frame_F, dtype=np.float64)      # (128, 8)
        SMR = np.asarray(SMR, dtype=np.float64)        # (NB, 8)
        num_subframes = 8
    else:
        X = np.asarray(frame_F, dtype=np.float64).reshape(-1, 1)  # (1024, 1)
        SMR = np.asarray(SMR, dtype=np.float64).reshape(-1)       # (NB,)
        num_subframes = 1

    # Scalefactor bands 
    wlow, whigh = load_scalefactor_bands(frame_type)
    NB = len(wlow)

    # Outputs 
    S = np.zeros((1024, 1), dtype=np.int32)

    if frame_type == "ESH":
        sfc = np.zeros((NB, 8), dtype=np.int32)
        G = np.zeros((1, 8), dtype=np.int32)
        T_all = compute_thresholds(X, SMR, wlow, whigh, frame_type)
    else:
        sfc = np.zeros((NB, 1), dtype=np.int32)
        G = np.int32(0)
        T_all = None

    # Process subframes 
    for sf in range(num_subframes):
        Xsf = X[:, sf]

        if frame_type == "ESH":
            T = T_all[:, sf]
        else:
            T = compute_thresholds(Xsf, SMR, wlow, whigh, frame_type)

        # Initial alpha
        alpha_hat = initial_alpha_estimate(Xsf)
        alpha = np.full(NB, alpha_hat, dtype=np.int32)

        # Iterative alpha optimization
        for b in range(NB):
            while True:
                Xb = Xsf[wlow[b]:whigh[b] + 1]
                Pe = _band_error_power(Xb, alpha[b])

                if Pe >= T[b]:
                    break

                candidate = alpha[b] + 1

                if b > 0 and abs(candidate - alpha[b - 1]) > MAX_SFC_DIFF:
                    break
                if b < NB - 1 and abs(alpha[b + 1] - candidate) > MAX_SFC_DIFF:
                    break

                alpha[b] = candidate

        # Final quantization
        alpha_k = build_alpha_per_coeff(alpha, wlow, whigh, len(Xsf))
        Ssf = quantize(Xsf, alpha_k).astype(np.int32)

        if frame_type == "ESH":
            S[sf * 128:(sf + 1) * 128, 0] = Ssf
            G[0, sf] = alpha[0]
            sfc[0, sf] = alpha[0]
            for b in range(1, NB):
                sfc[b, sf] = alpha[b] - alpha[b - 1]
        else:
            S[:, 0] = Ssf
            G = np.int32(alpha[0])
            sfc[0, 0] = alpha[0]
            for b in range(1, NB):
                sfc[b, 0] = alpha[b] - alpha[b - 1]

    return S, sfc, G
