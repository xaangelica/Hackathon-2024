# data_acquisition.py

import mne

def load_eeg_data(file_path):
    """
    Load EEG data from a .set file.
    
    Parameters:
    - file_path: str, path to the .set file.
    
    Returns:
    - raw: mne.io.Raw object containing EEG data.
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw
