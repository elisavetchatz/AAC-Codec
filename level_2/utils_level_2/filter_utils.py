import numpy as np
from numpy.polynomial.polynomial import Polynomial


def check_filter_stability(a):
    """
    Check if the inverse filter H_TNS^{-1} is stable.
    
    The inverse filter is 1 / (1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p)
    It is stable if all poles are inside the unit circle |z| < 1.
    
    Args:
        a: LPC coefficients [a1, a2, ..., ap]
    
    Returns:
        is_stable: True if filter is stable
    """
    # The inverse filter has denominator: 1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p
    # Convert to polynomial form: z^p - a1*z^(p-1) - a2*z^(p-2) - ... - ap
    
    p = len(a)
    
    # Polynomial coefficients in descending order of powers
    # For numpy.polynomial.polynomial, we need ascending order
    poly_coeffs = np.zeros(p + 1)
    if p > 0:
        poly_coeffs[0] = -a[-1]  # Constant term
    else:
        poly_coeffs[0] = 0

    for i in range(1, p):
        poly_coeffs[i] = -a[p - 1 - i]

    poly_coeffs[p] = 1.0  # Highest degree term
    
    # Find roots (poles)
    poly = Polynomial(poly_coeffs)
    roots = poly.roots()
    
    # Check if all roots are inside unit circle
    is_stable = np.all(np.abs(roots) < 1.0)
    
    return is_stable

def apply_tns_filter(X, a):
    """
    Apply TNS FIR filter: H_TNS(z) = 1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p
    
    Args:
        X: Input MDCT coefficients
        a: Quantized LPC coefficients
    
    Returns:
        Y: Filtered MDCT coefficients
    """
    X = np.asarray(X).flatten()
    N = len(X)
    p = len(a)
    
    Y = np.zeros(N)
    
    for k in range(N):
        Y[k] = X[k]
        for l in range(1, min(k + 1, p + 1)):
            Y[k] -= a[l - 1] * X[k - l]
    
    return Y

def apply_inverse_tns_filter(Y, a):
    """
    Apply inverse TNS filter: H_TNS^{-1}(z) = 1 / (1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p)
    
    This is an IIR filter that reverses the TNS encoding.
    
    Args:
        Y: TNS-filtered MDCT coefficients
        a: Quantized LPC coefficients
    
    Returns:
        X: Reconstructed MDCT coefficients
    """
    Y = np.asarray(Y).flatten()
    N = len(Y)
    p = len(a)
    
    X = np.zeros(N)
    
    for k in range(N):
        X[k] = Y[k]
        for l in range(1, min(k + 1, p + 1)):
            X[k] += a[l - 1] * X[k - l]
    
    return X