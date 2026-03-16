import numpy as np
import os
from scipy.io import loadmat

MAGIC_NUMBER = 0.4054
MQ = 8191


def load_scalefactor_bands(frame_type):
    """
    Load scalefactor band limits from TableB219.mat.
    According to the assignment, scalefactor bands
    are identical to psychoacoustic bands.
    """
    # Get the path to TableB219.mat in the workspace root
    module_dir = os.path.dirname(__file__)
    workspace_root = os.path.join(module_dir, '../..')
    table_path = os.path.join(workspace_root, 'TableB219.mat')
    tables = loadmat(table_path)

    if frame_type == "ESH":
        band_table = tables["B219b"]   # short windows (42 bands)
    else:
        band_table = tables["B219a"]   # long windows (69 bands)

    wlow = band_table[:, 2].astype(int)
    whigh = band_table[:, 3].astype(int)

    return wlow, whigh


def compute_thresholds(X, SMR, wlow, whigh, frame_type):
    """
    Compute hearing thresholds T(b) = P(b) / SMR(b) per scalefactor band.

    For non-ESH:
        X: (1024,) or (1024,1), SMR: (NB,)
        returns T: (NB,)

    For ESH:
        X: (128,8), SMR: (NB,8)
        returns T: (NB,8)
    """
    NB = len(wlow)
    X = np.asarray(X)
    SMR = np.asarray(SMR)

    if frame_type == "ESH":
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError(f"ESH expects X shape (128,8), got {X.shape}")
        if SMR.ndim != 2 or SMR.shape[1] != 8:
            raise ValueError(f"ESH expects SMR shape (NB,8), got {SMR.shape}")

        T = np.zeros((NB, 8), dtype=np.float64)
        for s in range(8):
            for b in range(NB):
                band_energy = np.sum(X[wlow[b]:whigh[b] + 1, s] ** 2)
                smr = SMR[b, s]
                T[b, s] = band_energy / smr if smr > 0 else 0.0
        return T

    # non-ESH
    X1 = X.reshape(-1)
    SMR1 = SMR.reshape(-1)
    T = np.zeros(NB, dtype=np.float64)
    for b in range(NB):
        band_energy = np.sum(X1[wlow[b]:whigh[b] + 1] ** 2)
        smr = SMR1[b]
        T[b] = band_energy / smr if smr > 0 else 0.0
    return T


def initial_alpha_estimate(X):
    """
    Initial estimate of scalefactor gain alpha_hat according to Eq. (14).
    Note: max is taken over ALL MDCT coefficients of the (sub)frame.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    absX = np.abs(X)
    if np.all(absX == 0):
        return 0

    max_pow = np.max(absX ** (3 / 4))
    alpha_hat = (16 / 3) * np.log2(max_pow / MQ)
    return int(np.floor(alpha_hat))


def build_alpha_per_coeff(alpha_band, wlow, whigh, N):
    """
    Expand band-wise alpha(b) values to coefficient-wise alpha(k).
    Works for N=1024 (long) or N=128 (short subframe).
    """
    alpha_k = np.zeros(N, dtype=np.float64)
    for b in range(len(alpha_band)):
        alpha_k[wlow[b]:whigh[b] + 1] = alpha_band[b]
    return alpha_k


def quantize(X, alpha_k):
    """
    Quantize MDCT coefficients according to Eq. (12).
    Uses truncation (round toward zero) for int[...].
    """
    X = np.asarray(X, dtype=np.float64)
    alpha_k = np.asarray(alpha_k, dtype=np.float64)
    return np.sign(X) * np.trunc(
        (np.abs(X) * (2 ** (-alpha_k / 4))) ** (3 / 4) + MAGIC_NUMBER
    )


def dequantize(S, alpha_k):
    """
    Dequantize MDCT coefficients according to Eq. (13).
    """
    S = np.asarray(S, dtype=np.float64)
    alpha_k = np.asarray(alpha_k, dtype=np.float64)
    return np.sign(S) * (np.abs(S) ** (4 / 3)) * (2 ** (alpha_k / 4))

def band_error_power(Xb, alpha_b):
    """
    Quantization error power for one scalefactor band.
    """
    alpha_k = np.full_like(Xb, alpha_b, dtype=np.float64)
    Sb = quantize(Xb, alpha_k)
    Xhat = dequantize(Sb, alpha_k)
    return np.sum((Xb - Xhat) ** 2)


def _band_error_power(Xb, alpha_b):
    """
    Quantization error power for one scalefactor band.
    Alias for band_error_power to maintain compatibility.
    """
    return band_error_power(Xb, alpha_b)