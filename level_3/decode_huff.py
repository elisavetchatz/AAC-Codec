from huff_utils import load_LUT, decode_huff


def aac_huffman_decode(huff_S, cb_S, huff_sfc, cb_sfc):
    """
    Huffman decoding stage for AAC quantizer outputs.

    Decodes:
    - Quantized MDCT coefficients
    - Scalefactors

    Args:
        huff_S (str): Huffman bitstream for MDCT coefficients
        cb_S (int): Huffman codebook used for MDCT coefficients
        huff_sfc (str): Huffman bitstream for scalefactors
        cb_sfc (int): Huffman codebook used for scalefactors

    Returns:
        S (ndarray): Decoded quantized MDCT coefficients (1024x1)
        sfc (ndarray): Decoded scalefactors (1D array)
    """

    # Load Huffman lookup tables
    huff_LUT = load_LUT()

    # Decode MDCT coefficients 
    S_dec = decode_huff(huff_S, huff_LUT[cb_S])

    # Decode scalefactors
    sfc_dec = decode_huff(huff_sfc, huff_LUT[cb_sfc])

    return np.array(S_dec).reshape(-1, 1), np.array(sfc_dec)
