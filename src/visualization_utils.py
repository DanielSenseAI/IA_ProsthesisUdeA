import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import savgol_filter
import src.db_utils as db_utils
import src.preprocessing_utils as prep_utils
from src.config import DATABASE_INFO

def plot_data(filtered_emg_data, restimulus_data, grasp_number=None, interactive=False, frequency=None):
    emg_df = pd.DataFrame(filtered_emg_data, columns=[f'Channel {i+1}' for i in range(filtered_emg_data.shape[1])])

    if frequency is not None:
        emg_df = prep_utils.add_time(emg_df, frequency)
        x_axis = 'Time (s)'
    else:
        x_axis = 'Sample'

    if grasp_number is None:
        title = 'EMG Data for All Restimuli'
    else:
        title = f'EMG Data for Restimulus {grasp_number}'
    
    if interactive:
        fig = px.line(emg_df, x=x_axis, y=emg_df.columns[:-1], title=title)
        fig.update_layout(xaxis_title=x_axis, yaxis_title='Amplitude')
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=(18, 6))  # Create figure and axis
        emg_df.plot(x=x_axis, title=title, ax=ax)
        ax.set_xlabel(x_axis, fontsize=10)  # Adjust the font size of the x-axis label
        ax.set_ylabel('Amplitude', fontsize=10)  # Adjust the font size of the y-axis label

        plot_stimulus(ax, emg_df, restimulus_data)
        ax.legend(loc='upper right', fontsize=6)
        plt.show()

def plot_stimulus(ax, emg_Data, restimulus_data):
    start_index, end_index = prep_utils.get_transition_indexes(restimulus_data)
    #print(f'Start index: {start_index}')
    #print(f'End index: {end_index}')

    start_times = emg_Data['Time (s)'].iloc[start_index].values  # Replace with actual mapping logic if needed
    end_times = emg_Data['Time (s)'].iloc[end_index].values  # Replace with actual mapping logic if needed

    # Add vertical lines for transitions
    for i, time in enumerate(start_times):
        ax.axvline(
            x=time,
            color='red',
            linestyle='--',
            linewidth=0.8,
            label='Start Transition' if i == 0 else ""  # Label only the first line for legend clarity
        )

    for i, time in enumerate(end_times):
        ax.axvline(
            x=time,
            color='blue',
            linestyle='--',
            linewidth=0.8,
            label='End Transition' if i == 0 else ""  # Label only the first line for legend clarity
        )

def plot_emg_data(database, mat_file, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False, addFourier = False):
    try:
        emg_data, restimulus_data = db_utils.extract_data(mat_file, use_stimulus)
    except KeyError as e:
        print(f"KeyError in extract_data: {e}")
        raise

    if time == True:
        try:
            frequency = DATABASE_INFO[database]['frequency']
        except KeyError as e:
            print(f"KeyError accessing DATABASE_INFO: {e}")
            raise
    else:
        frequency = None

    if emg_data is None or restimulus_data is None:
        return

    try:
        filtered_emg_data, filtered_restimulus_data = db_utils.filter_data(emg_data, restimulus_data, grasp_number, include_rest, padding = padding)
    except KeyError as e:
        print(f"KeyError in filter_data: {e}")
        raise

    # Debugging: Print the shapes of the filtered data
    print(f"Filtered EMG data shape: {filtered_emg_data.shape}")
    print(f"Filtered restimulus data shape: {filtered_restimulus_data.shape}")
    print(f"test time: {len(filtered_emg_data) / frequency} seconds")

    # Check if filtered data is None
    if filtered_emg_data is None or filtered_restimulus_data is None:
        raise ValueError("Filtered data is None")

    plot_data(filtered_emg_data, filtered_restimulus_data, grasp_number, interactive, frequency)

    if addFourier:
        plot_fourier_transform_with_envelope(filtered_emg_data, frequency)

def plot_emg_dataframe(database, emg_data, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False, addFourier = False):
    if time == True:
        try:
            frequency = DATABASE_INFO[database]['frequency']
        except KeyError as e:
            print(f"KeyError accessing DATABASE_INFO: {e}")
            raise
    else:
        frequency = None

    if emg_data is None or emg_data['stimulus'] is None:
        return

    try:
        filtered_emg_data = db_utils.filter_data_pandas(emg_data, grasp_number, include_rest=include_rest, padding = padding)
    except KeyError as e:
        print(f"KeyError in filter_data: {e}")
        raise

    # Debugging: Print the shapes of the filtered data
    print(f"Filtered EMG data shape: {filtered_emg_data.shape}")
    print(f"test time: {len(filtered_emg_data) / frequency} seconds")

    # Check if filtered data is None
    if filtered_emg_data is None:
        raise ValueError("Filtered data is None")

    plot_data(prep_utils.extract_emg_channels(filtered_emg_data), filtered_emg_data['relabeled'], grasp_number, interactive, frequency)

    if addFourier:
        plot_fourier_transform_with_envelope(filtered_emg_data, frequency)


def plot_fourier_transform(emg_data, frequency, start_freq=0, end_freq=600):
    # Compute the Fourier transform of the filtered EMG data
    fourier_data = np.fft.fft(emg_data, axis=0)
    freqs = np.fft.fftfreq(emg_data.shape[0], d=1/frequency)

    # Filter out negative frequencies
    positive_freqs = freqs > 0
    fourier_data = fourier_data[positive_freqs]
    freqs = freqs[positive_freqs]

    # Apply frequency boundaries
    if end_freq is None:
        end_freq = freqs[-1]
    freq_mask = (freqs >= start_freq) & (freqs <= end_freq)
    fourier_data = fourier_data[freq_mask]
    freqs = freqs[freq_mask]

    # Plot the envelope of the Fourier transform
    plt.figure(figsize=(18, 6))
    for i in range(fourier_data.shape[1]):
        plt.plot(freqs, np.abs(fourier_data[:, i]), label=f'Channel {i + 1}')
    
    plt.title('Fourier Transform of EMG Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    plt.show()

def plot_fourier_transform_with_envelope(emg_data, frequency, start_freq=0, end_freq=600, window_length=120, polyorder=3):
    """
    Plots the Fourier transform of EMG data with a smoothed envelope.
    
    Parameters:
        emg_data (np.ndarray): EMG data array with shape (samples, channels).
        frequency (float): Sampling frequency in Hz.
        start_freq (float): Start frequency for plotting, default is 0 Hz.
        end_freq (float): End frequency for plotting, default is Nyquist frequency.
        window_length (int): Window length for Savitzky-Golay filter (must be odd).
        polyorder (int): Polynomial order for Savitzky-Golay filter.
    """
    # Compute the Fourier transform of the filtered EMG data
    fourier_data = np.fft.fft(emg_data, axis=0)
    freqs = np.fft.fftfreq(emg_data.shape[0], d=1/frequency)

    # Filter out negative frequencies
    positive_freqs = freqs > 0
    fourier_data = fourier_data[positive_freqs]
    freqs = freqs[positive_freqs]

    # Apply frequency boundaries
    if end_freq is None:
        end_freq = freqs[-1]
    freq_mask = (freqs >= start_freq) & (freqs <= end_freq)
    fourier_data = fourier_data[freq_mask]
    freqs = freqs[freq_mask]

    # Compute the magnitude of the Fourier transform
    magnitude = np.abs(fourier_data)

    # Apply Savitzky-Golay filter to smooth the envelope
    smoothed_magnitude = savgol_filter(magnitude, window_length=window_length, polyorder=polyorder, axis=0)

    # Plot the smoothed envelope
    plt.figure(figsize=(18, 6))
    for i in range(smoothed_magnitude.shape[1]):
        plt.plot(freqs, smoothed_magnitude[:, i], label=f'Channel {i + 1}')
    
    plt.title('Smoothed Envelope of Fourier Transform of EMG Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    plt.show()