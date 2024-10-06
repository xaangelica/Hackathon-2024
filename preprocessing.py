import mne
from mne.preprocessing import ICA

def preprocess_eeg(raw):
    """
    Preprocess EEG data by filtering and re-referencing.
    
    Parameters:
    - raw: mne.io.Raw object containing EEG data.
    
    Returns:
    - raw: mne.io.Raw object after preprocessing.
    """
    # Band-pass filter between 1 and 50 Hz
    raw.filter(l_freq=1., h_freq=50.)

    # Set average reference
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()

    return raw

def remove_artifacts(raw):
    """
    Remove artifacts using Independent Component Analysis (ICA).
    
    Parameters:
    - raw: mne.io.Raw object containing EEG data.
    
    Returns:
    - raw_corrected: mne.io.Raw object after artifact removal.
    """
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    raw_corrected = ica.apply(raw.copy())
    return raw_corrected
