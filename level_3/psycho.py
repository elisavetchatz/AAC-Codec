import numpy as np

from utils_level_3.psycho_utils import (get_spreading_tables, process_frame_fft, 
                                         compute_predictions, compute_predictability,
                                         compute_band_energy_predictability,
                                         apply_spreading_function, compute_tonality_index,
                                         compute_snr)

# True to print psychoacoustic model statistics
DEBUG = False


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):
    """
    Psychoacoustic model implementation for one channel.

    Args:
        frame_T (array): Current frame in time domain
        frame_type (str): Frame type. Can be 'OLS', 'LSS', 'ESH', 'LPS'.
        frame_T_prev_1 (array): Previous frame of frame_T in the same channel
        frame_T_prev_2 (array): Frame before the previous frame of frame_T in the same channel
    
    Returns:
        SMR (array): Signal to Mask Ratio
                    Dimensions: 42x8 for EIGHT_SHORT_SEQUENCE frames, 69x1 for all other types
    """
    
    tables = get_spreading_tables()
    spreading_long = tables['spreading_long']
    spreading_short = tables['spreading_short']
    bval_long = tables['bval_long']
    bval_short = tables['bval_short']
    wlow_long = tables['wlow_long']
    whigh_long = tables['whigh_long']
    wlow_short = tables['wlow_short']
    whigh_short = tables['whigh_short']
    
    if frame_type == 'ESH':

        num_bands = len(bval_short)
        num_windows = 8
        SMR = np.zeros((num_bands, num_windows))
        
        all_subframes = []
        
        for frame in [frame_T_prev_2, frame_T_prev_1, frame_T]:
            for subframe_idx in range(num_windows):
                start_idx = subframe_idx * 128
                end_idx = start_idx + 256
                subframe = frame[start_idx:end_idx]

                r, f = process_frame_fft(subframe)
                all_subframes.append({'r': r, 'f': f})
        
        # Now compute predictions and predictability for each current subframe in one loop
        predictabilities = []
        for i in range(num_windows):
            current_idx = 16 + i  # Current subframe index in all_subframes
            prev_1_idx = current_idx - 1  # Previous subframe
            prev_2_idx = current_idx - 2  # Previous-previous subframe
            
            # Get current values
            r_current = all_subframes[current_idx]['r']
            f_current = all_subframes[current_idx]['f']
            
            # Get previous values for predictions
            r_prev_2 = all_subframes[prev_2_idx]['r']
            f_prev_2 = all_subframes[prev_2_idx]['f']
            r_prev_1 = all_subframes[prev_1_idx]['r']
            f_prev_1 = all_subframes[prev_1_idx]['f']
            
            # Compute predictions
            rpred, fpred = compute_predictions(r_prev_2, f_prev_2, r_prev_1, f_prev_1)
            
            # Compute predictability measure
            c = compute_predictability(r_current, f_current, rpred, fpred)
            predictabilities.append(c)
        
        # Step 5: Compute energy and weighted predictability for each subframe
        e_bands_all = []
        c_bands_all = []
        cb_all = []
        en_all = []
        tb_all = []
        SNR_all = []
        for i in range(num_windows):
            r_current = all_subframes[16 + i]['r']
            c = predictabilities[i]
            e_bands, c_bands = compute_band_energy_predictability(r_current, c, wlow_short, whigh_short)
            e_bands_all.append(e_bands)
            c_bands_all.append(c_bands)
            
            # Step 6: Apply spreading function and normalize
            cb, en = apply_spreading_function(e_bands, c_bands, spreading_short)
            cb_all.append(cb)
            en_all.append(en)
            
            # Step 7: Compute tonality index
            tb = compute_tonality_index(cb)
            tb_all.append(tb)
            
            # Step 8: Compute required SNR based on tonality
            SNR_dB = compute_snr(tb)
            SNR_all.append(SNR_dB)
        
        # Debug info: Show statistics for first subframe
        if DEBUG:
            print("\n=== Short Frame Psychoacoustic Model Statistics (Subframe 0) ===")
            tb_0 = tb_all[0]
            SNR_0 = SNR_all[0]
            en_0 = en_all[0]
            print(f"Tonality index (tb): min={tb_0.min():.3f}, max={tb_0.max():.3f}, mean={tb_0.mean():.3f}")
            print(f"Required SNR (dB): min={SNR_0.min():.1f}, max={SNR_0.max():.1f}, mean={SNR_0.mean():.1f}")
            print(f"Normalized energy (en): min={en_0.min():.2e}, max={en_0.max():.2e}")
        
        # TODO: Continue with step 9 of psychoacoustic model for short frames
        
    else:  # OLS, LSS, LPS (long frames)
        num_bands = len(bval_long)
        SMR = np.zeros((num_bands, 1))
        
        # Step 2: Process the 3 frames (current + 2 previous) - each is 2048 samples
        r_prev_2, f_prev_2 = process_frame_fft(frame_T_prev_2)
        r_prev_1, f_prev_1 = process_frame_fft(frame_T_prev_1)
        r_current, f_current = process_frame_fft(frame_T)
        
        # Step 3: Compute predictions using the 2 previous frames
        rpred, fpred = compute_predictions(r_prev_2, f_prev_2, r_prev_1, f_prev_1)
        
        # Step 4: Compute predictability measure c(w)
        c = compute_predictability(r_current, f_current, rpred, fpred)
        
        # Step 5: Compute energy and weighted predictability for each band
        e_bands, c_bands = compute_band_energy_predictability(r_current, c, wlow_long, whigh_long)
        
        # Step 6: Apply spreading function and normalize
        cb, en = apply_spreading_function(e_bands, c_bands, spreading_long)
        
        # Step 7: Compute tonality index
        tb = compute_tonality_index(cb)
        
        # Step 8: Compute required SNR based on tonality
        # TMN (Tone Masking Noise) = 18 dB: tonal signals mask noise easily
        # NMT (Noise Masking Tone) = 6 dB: noise masks tones with lower SNR requirement
        SNR_dB = compute_snr(tb)
        
        # Debug info: Show statistics of psychoacoustic parameters
        if DEBUG:
            print("\n=== Long Frame Psychoacoustic Model Statistics ===")
            print(f"Tonality index (tb): min={tb.min():.3f}, max={tb.max():.3f}, mean={tb.mean():.3f}")
            print(f"Required SNR (dB): min={SNR_dB.min():.1f}, max={SNR_dB.max():.1f}, mean={SNR_dB.mean():.1f}")
            print(f"Normalized energy (en): min={en.min():.2e}, max={en.max():.2e}")
            print(f"Interpretation:")
            print(f"  - Tonal bands (tb>{0.7}): {np.sum(tb > 0.7)} / {num_bands}")
            print(f"  - Noisy bands (tb<{0.3}): {np.sum(tb < 0.3)} / {num_bands}")
        
        # TODO: Continue with step 9 of psychoacoustic model for long frames
    
    return SMR
