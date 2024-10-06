import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from data_acquisition import get_data_chunks

class RealTimeEEGVisualizer:
    def __init__(self, raw, chunk_size=100):
        """
        Initialize the real-time EEG visualizer.
        
        Parameters:
        - raw: mne.io.Raw object containing EEG data.
        - chunk_size: int, number of samples per chunk to display.
        """
        self.raw = raw
        self.chunk_size = chunk_size
        self.chunk_generator = None
        
        # Get all channels
        self.channels_to_display = np.arange(raw.info['nchan'])
        self.sampling_freq = raw.info['sfreq']  # Get the sampling frequency

        # Set up PyQtGraph window without axes or labels
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Visualization")
        self.plots = []

        # Create a plot for each channel without axes
        for i in range(len(self.channels_to_display)):
            plot = self.win.addPlot(row=i, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            curve = plot.plot()
            self.plots.append(curve)

        self.data = {ch: np.array([]) for ch in self.channels_to_display}  # Store data per channel

    def update(self):
        """
        Update the visualization with the next chunk of EEG data.
        """
        try:
            chunk = next(self.chunk_generator)
            
            # Update each channel individually
            for idx, channel in enumerate(self.channels_to_display):
                channel_data = chunk[channel, :]  # Extract data for the specific channel
                self.data[channel] = np.concatenate((self.data[channel], channel_data)).flatten()

                # Keep only the last 500 samples for visualization
                if self.data[channel].size > int(self.sampling_freq):  # Adjust to show 1 second of data
                    self.data[channel] = self.data[channel][-int(self.sampling_freq):]

                # Update the curve with x and y data
                self.plots[idx].setData(self.data[channel])

        except StopIteration:
            print("No more data to display.")

    def start(self):
        """
        Start the visualization loop.
        """
        # Initialize chunk generator
        self.chunk_generator = get_data_chunks(self.raw, self.chunk_size)

        # Set up a timer to update the visualization periodically
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(int(1000 / self.sampling_freq))  # Update based on the sampling frequency
        
        # Start the Qt event loop
        QtWidgets.QApplication.instance().exec_()
