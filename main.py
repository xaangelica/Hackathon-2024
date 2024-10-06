import mne
import matplotlib.pyplot as plt

from data_acquisition import load_eeg_data
from preprocessing import preprocess_eeg, remove_artifacts
from feature_extraction import extract_band_powers

def main():
    # Step 1: Data Acquisition
    eeg_file_path = "8-16.set" 
    raw = load_eeg_data(eeg_file_path)
    
    # Step 2: Preprocessing
    raw = preprocess_eeg(raw)
    raw = remove_artifacts(raw)
    
    # Step 3: Segment Data (Epoching)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0))
    
    # Step 4: Feature Extraction
    features = extract_band_powers(epochs)
    print(f"Extracted features shape: {features.shape}")

    # Forzar visualización interactiva
    plt.ion()
    
    # Visualize the epochs
    epochs.plot()

    # Pausar la ejecución para mantener la ventana abierta
    plt.show(block=True)

if __name__ == "__main__":
    main()
