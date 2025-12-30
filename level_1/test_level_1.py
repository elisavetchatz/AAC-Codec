"""
Test script for AAC Level 1 encoding/decoding pipeline
Designed to run in CI/CD environment
"""
import sys
import os
import numpy as np
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aac_coder_1 import aac_coder_1
from i_aac_coder_1 import i_aac_coder_1


def test_level_1():
    """
    Test AAC Level 1 encoding and decoding.
    Returns True if test passes, False otherwise.
    """
    print("=" * 70)
    print("AAC LEVEL 1 - AUTOMATED TEST")
    print("=" * 70)
    
    # Test configuration
    filename_in = "../LicorDeCalandraca.wav"
    filename_out = "../output.wav"
    min_snr_threshold = 30.0  # Minimum acceptable SNR in dB
    
    try:
        # Check input file exists
        if not os.path.exists(filename_in):
            print(f"ERROR: Input file '{filename_in}' not found!")
            return False
        
        print(f"\n[1/5] Reading input file: {filename_in}")
        x_original, fs = sf.read(filename_in)
        
        # Ensure float format
        if x_original.dtype == np.int16:
            x_original = x_original.astype(np.float32) / 32768.0
        elif x_original.dtype == np.int32:
            x_original = x_original.astype(np.float32) / 2147483648.0
        
        print(f"      Sample rate: {fs} Hz")
        print(f"      Duration: {len(x_original) / fs:.2f} seconds")
        print(f"      Channels: {x_original.shape[1] if x_original.ndim > 1 else 1}")
        
        # Encode
        print(f"\n[2/5] Encoding with AAC Level 1...")
        aac_seq_1 = aac_coder_1(filename_in)
        print(f"      Encoded {len(aac_seq_1)} frames")
        
        # Decode
        print(f"\n[3/5] Decoding...")
        x_decoded = i_aac_coder_1(aac_seq_1, filename_out)
        
        # Verify output file
        if not os.path.exists(filename_out):
            print(f"ERROR: Output file '{filename_out}' was not created!")
            return False
        print(f"      Output file created: {filename_out}")
        
        # Calculate SNR
        print(f"\n[4/5] Calculating SNR...")
        
        # Ensure same length
        if len(x_original) != len(x_decoded):
            min_len = min(len(x_original), len(x_decoded))
            x_original = x_original[:min_len]
            x_decoded = x_decoded[:min_len]
            print(f"      (Trimmed to {min_len} samples)")
        
        signal_power = np.sum(x_original ** 2)
        noise_power = np.sum((x_original - x_decoded) ** 2)
        
        if noise_power > 0:
            SNR = 10 * np.log10(signal_power / noise_power)
        else:
            SNR = float('inf')
        
        print(f"      SNR: {SNR:.2f} dB")
        
        # Validate SNR
        print(f"\n[5/5] Validating results...")
        print(f"      Threshold: {min_snr_threshold:.2f} dB")
        
        if SNR < min_snr_threshold:
            print(f"      FAIL: SNR {SNR:.2f} dB is below threshold {min_snr_threshold:.2f} dB")
            return False
        
        print(f"      PASS: SNR {SNR:.2f} dB exceeds threshold")
        
        print("\n" + "=" * 70)
        print("TEST PASSED ✓")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception:")
        print(f"       {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_level_1()
    sys.exit(0 if success else 1)
