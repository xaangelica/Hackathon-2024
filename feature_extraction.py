# feature_extraction.py

import numpy as np
from mne.time_frequency import psd_array_welch

def extract_band_powers(epochs):
    """
    Extract average band powers from EEG epochs.
    
    Parameters:
    - epochs: mne.Epochs object containing segmented EEG data.
    
    Returns:
    - band_powers: dict with average band powers per frequency band.
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    # Extract the data and the sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    
    # Compute power spectral density (PSD) for each epoch
    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=50, n_fft=256)

    band_powers = {}
    for band in bands:
        fmin, fmax = bands[band]
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = psds[:, :, idx_band].mean(axis=-1)
    
    # Flatten band powers for easier handling in machine learning
    flattened_features = np.concatenate([band_powers[band] for band in bands], axis=1)
    
    return flattened_features
