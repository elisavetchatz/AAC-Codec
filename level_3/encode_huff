import numpy as np
from huff_utils import load_LUT, encode_huff


def aac_huffman_encode(S, sfc):
    """
    Huffman encoding stage for AAC quantizer outputs.

    Encodes:
    - Quantized MDCT coefficients S
    - Scalefactors sfc

    According to the assignment:
    - MDCT coefficients use automatically selected Huffman codebooks
    - Scalefactors are encoded using codebook 11

    Args:
        S (ndarray): Quantized MDCT coefficients (1024x1)
        sfc (ndarray): Scalefactors (NBx1 or NBx8)

    Returns:
        huff_S (str): Huffman bitstream for MDCT coefficients
        cb_S (int): Huffman codebook used for MDCT coefficients
        huff_sfc (str): Huffman bitstream for scalefactors
        cb_sfc (int): Huffman codebook used for scalefactors (always 11)
    """

    # Load Huffman lookup tables
    huff_LUT = load_LUT()

    # Encode MDCT coefficients
    huff_S, cb_S = encode_huff(
        coeff_sec = S.flatten(),
        huff_LUT_list = huff_LUT
    )

    # Encode scalefactors 
    huff_sfc, cb_sfc = encode_huff(
        coeff_sec = sfc.flatten(),
        huff_LUT_list = huff_LUT,
        force_codebook = 11
    )

    return huff_S, cb_S, huff_sfc, cb_sfc
