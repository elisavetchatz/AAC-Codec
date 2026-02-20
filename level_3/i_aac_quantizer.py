import numpy as np

from utils_level_3.quantizer_utils import (
    load_scalefactor_bands,
    build_alpha_per_coeff,
    dequantize
)
def i_aac_quantizer(S, sfc, G, frame_type):
    """
    Inverse quantizer stage implementation (reverses aac_quantizer()).

    Args:
        S (array): Array of quantized MDCT coefficient symbols for current frame
                  Dimensions: 1024×1 (for all frame types)
        sfc (array): Scalefactor values for each Scalefactor band
                    Dimensions: NB×8 for EIGHT_SHORT_SEQUENCE frames (where NB is number of bands),
                               NB×1 for all other types
        G (array or float): Global gain of current frame
                           Dimensions: 1×8 for EIGHT_SHORT_SEQUENCE or single value for all other cases
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
    
    Returns:
        frame_F (array): MDCT coefficients
                        Dimensions: 128×8 for EIGHT_SHORT_SEQUENCE, 1024×1 otherwise
    """
    # Flatten symbols
    S = S.flatten()

    # Load scalefactor band limits
    wlow, whigh = load_scalefactor_bands(frame_type)
    NB = len(wlow)

    # EIGHT_SHORT_SEQUENCE
    if frame_type == "ESH":

        frame_F = np.zeros((128, 8))

        for sf in range(8):

            alpha_b = np.zeros(NB)
            alpha_b[0] = G[0, sf]  # Use Global Gain

            for b in range(1, NB):
                if b < sfc.shape[0]:  # Safety check
                    alpha_b[b] = alpha_b[b-1] + sfc[b, sf]

            alpha_k = build_alpha_per_coeff(alpha_b, wlow, whigh, 128)

            Ssf = S[sf * 128:(sf + 1) * 128]

            frame_F[:, sf] = dequantize(Ssf, alpha_k)

        return frame_F

    # LONG FRAMES (OLS, LSS, LPS)
    else:

        sfc_flat = sfc.flatten()
        alpha_b = np.zeros(NB)
        alpha_b[0] = G  

        for b in range(1, NB):
            if b < len(sfc_flat):  # Safety check
                alpha_b[b] = alpha_b[b-1] + sfc_flat[b]
        
        alpha_k = build_alpha_per_coeff(alpha_b, wlow, whigh, 1024)

        frame_F = dequantize(S, alpha_k)

    return frame_F
