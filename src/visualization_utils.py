import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import savgol_filter, convolve
from scipy.ndimage import gaussian_filter1d

import src.db_utils as db_utils
import src.preprocessing_utils as prep_utils
from src.config import DATABASE_INFO
from src.preprocessing_utils import get_transition_indexes
from src.preprocessing_utils import extract_emg_channels

def plot_data(filtered_emg_data, restimulus_data, grasp_number=None, interactive=False, frequency=None, title=None, sort_channels=False):
    emg_df = pd.DataFrame(filtered_emg_data, columns=[f'Channel {i+1}' for i in range(filtered_emg_data.shape[1])])

    if frequency is not None:
        emg_df = prep_utils.add_time(emg_df, frequency)
        x_axis = 'Time (s)'
    else:
        x_axis = 'Sample'

    if title is None:
        if grasp_number is None:
            title = 'EMG Data for All Restimuli'
        else:
            title = f'EMG Data for Restimulus {grasp_number}'
    else: 
        title = title

    if interactive:
        fig = px.line(emg_df, x=x_axis, y=emg_df.columns[:-1], title=title)
        fig.update_layout(xaxis_title=x_axis, yaxis_title='Amplitude')
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=(18, 6))  # Create figure and axis
        if sort_channels:
            # Sort channels by their maximum absolute amplitude
            min_amplitudes = emg_df.abs().min()
            sorted_columns = min_amplitudes.sort_values().index
            emg_df = emg_df[sorted_columns]

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

def plot_emg_data(database, mat_file, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False, addFourier = False, title = None):
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

    plot_data(filtered_emg_data, filtered_restimulus_data, grasp_number, interactive, frequency, title)

    if addFourier:
        plot_fourier_transform_with_envelope(filtered_emg_data, frequency)


def plot_emg_dataframe(database, emg_data, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False, addFourier = False, length = 0.0, fourier_sigma = 25): 
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
        
        if length > 0.01:
            final_time = filtered_emg_data['Time (s)'].iloc[0] + length
            filtered_emg_data = filtered_emg_data[filtered_emg_data['Time (s)'] < final_time]
        
        filtered_restimulus_data = filtered_emg_data[['relabeled']]
        filtered_emg_data = prep_utils.extract_emg_channels(filtered_emg_data)

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
        plot_fourier_transform_with_envelope(filtered_emg_data, frequency, sigma=fourier_sigma)


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


def compute_fourier_transform(emg_data, frequency):
    """Computes the Fourier Transform and corresponding frequencies."""
    fourier_data = np.fft.fft(emg_data, axis=0)
    freqs = np.fft.fftfreq(emg_data.shape[0], d=1/frequency)
    return fourier_data, freqs


def filter_frequencies(fourier_data, freqs, start_freq, end_freq, remove_zero_freq=True, zero_band=3.0):
    """
    Removes the 0 Hz frequency component (optionally within a small range) and selects the desired frequency range.
    
    Parameters:
        fourier_data (np.ndarray): Fourier-transformed data.
        freqs (np.ndarray): Corresponding frequency bins.
        start_freq (float): Lower bound of frequency range.
        end_freq (float): Upper bound of frequency range.
        remove_zero_freq (bool): If True, removes frequencies within the `zero_band` around 0 Hz.
        zero_band (float): Width of the exclusion zone around 0 Hz (default 1 Hz).
        
    Returns:
        Filtered fourier_data and frequency array.
    """
    if remove_zero_freq:
        valid_freqs = (freqs < -zero_band) | (freqs > zero_band)  # Removes frequencies within [-zero_band, zero_band]
    else:
        valid_freqs = np.ones_like(freqs, dtype=bool)  # Keep all frequencies

    fourier_data, freqs = fourier_data[valid_freqs], freqs[valid_freqs]

    # Apply frequency range selection
    freq_mask = (freqs >= start_freq) & (freqs <= end_freq)
    return fourier_data[freq_mask], freqs[freq_mask]



def apply_smoothing(magnitude, window_length=101, polyorder=3, sigma=9):
    """Applies Savitzky-Golay and Gaussian smoothing."""
    smoothed = savgol_filter(magnitude, window_length=window_length, polyorder=polyorder, axis=0)
    return gaussian_filter1d(smoothed, sigma=sigma, axis=0)


def compute_frequency_metrics(freqs, smoothed_magnitude):
    """Computes max, median, and center frequencies for each channel."""
    max_freqs, median_freqs, center_freqs = [], [], []
    
    for i in range(smoothed_magnitude.shape[1]):
        max_freq = freqs[np.argmax(smoothed_magnitude[:, i])]
        max_freqs.append(max_freq)

        cumulative_power = np.cumsum(smoothed_magnitude[:, i])
        median_freq = freqs[np.searchsorted(cumulative_power, cumulative_power[-1] / 2)]
        median_freqs.append(median_freq)

        center_freq = np.sum(freqs * smoothed_magnitude[:, i]) / np.sum(smoothed_magnitude[:, i])
        center_freqs.append(center_freq)

    return max_freqs, median_freqs, center_freqs


def plot_fourier_transform_with_envelope(emg_data, frequency, start_freq=5, end_freq=600, 
                                         window_length=101, polyorder=3, sigma=9, 
                                         print_max=True, remove_zero_freq=True):
    """
    Computes and plots the Fourier transform of EMG data with a smoothed envelope.
    """
    # Compute FFT
    fourier_data, freqs = compute_fourier_transform(emg_data, frequency)
    
    # Filter frequencies
    #fourier_data, freqs = filter_frequencies(fourier_data, freqs, start_freq, end_freq, remove_zero_freq)
    # Apply frequency boundaries
    if end_freq is None:
        end_freq = freqs[-1]
    freq_mask = (freqs >= start_freq) & (freqs <= end_freq)
    fourier_data = fourier_data[freq_mask]
    freqs = freqs[freq_mask]

    # Compute magnitude
    magnitude = np.abs(fourier_data)

    # Apply smoothing
    smoothed_magnitude = apply_smoothing(magnitude, window_length, polyorder, sigma)

    # Compute frequency metrics
    max_freqs, median_freqs, center_freqs = compute_frequency_metrics(freqs, smoothed_magnitude)

    # Plot
    plt.figure(figsize=(18, 6))
    for i in range(smoothed_magnitude.shape[1]):
        plt.plot(freqs, smoothed_magnitude[:, i], label=f'Channel {i + 1}')
        if print_max:
            print(f"{i + 1}: Max= {max_freqs[i]:.2f} Hz, Med= {median_freqs[i]:.2f} Hz, Cen= {center_freqs[i]:.2f} Hz")

    plt.title('Smoothed Envelope of Fourier Transform of EMG Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize=6)
    plt.xticks(np.arange(start_freq, end_freq + 1, step=20))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return max_freqs, median_freqs, center_freqs


def get_transition_indexes(restimulus_data):
    """
    Identifica los índices donde hay cambios en el restimulus.

    Parameters:
    - restimulus_data (np.ndarray): Arreglo de estímulos.

    Returns:
    - list: Índices donde ocurren cambios en el estímulo.
    """
    transitions = np.where(np.diff(restimulus_data) != 0)[0] + 1
    return transitions.tolist()


def calculate_stimulus_times(emg_data: pd.DataFrame, restimulus_data: np.ndarray, frequency: float) -> pd.DataFrame:
    """
    Calculates stimulus start and end times based on restimulus data and computes
    the average activation and non-activation times.

    Parameters:
    - emg_data (pd.DataFrame): DataFrame containing the EMG signals with a 'time' column (if calculated).
    - restimulus_data (np.ndarray): Stimulus data array indicating the transitions.
    - frequency (float): Sampling frequency in Hz.

    Returns:
    - pd.DataFrame: A DataFrame with the columns:
        - 'Stimulus': stimulus number.
        - 'Start_Time': Stimulus start time in seconds.
        - 'End_Time': Stimulus end time in seconds.
        - 'Duration': Duration of the stimulus (activation time).
    - dict: A dictionary with average activation and non-activation times:
        - 'Average_Activation_Time': Average duration of stimuli in seconds.
        - 'Average_Non_Activation_Time': Average duration of rest periods in seconds.
    """
    # Obtener los índices de transición
    transition_indexes = get_transition_indexes(restimulus_data)

    # Inicializar una lista para almacenar resultados
    stimulus_times = []
    activation_durations = []  # Para almacenar duraciones de activación
    non_activation_durations = []  # Para almacenar duraciones de no activación

    # Calcular el tiempo correspondiente a cada índice basado en la frecuencia de muestreo
    for i in range(0, len(transition_indexes) - 1, 2):
        start_idx = transition_indexes[i]
        end_idx = transition_indexes[i + 1]

        start_time = start_idx / frequency  # Tiempo de inicio en segundos
        end_time = end_idx / frequency      # Tiempo de fin en segundos
        stimulus_number = restimulus_data[start_idx]  # Identificar el número del estímulo
        duration = end_time - start_time   # Duración de la activación

        # Almacenar resultados
        stimulus_times.append({
            'Stimulus': stimulus_number,
            'Start_Time': start_time,
            'End_Time': end_time,
            'Duration': duration
        })
        activation_durations.append(duration)

        # Calcular tiempo de no activación si no es el último estímulo
        if i + 2 < len(transition_indexes):
            next_start_idx = transition_indexes[i + 2]
            rest_duration = (next_start_idx - end_idx) / frequency  # Tiempo de reposo
            non_activation_durations.append(rest_duration)

    # Convertir resultados en DataFrame
    stimulus_times_df = pd.DataFrame(stimulus_times)

    # Calcular promedios
    avg_activation_time = np.mean(activation_durations) if activation_durations else 0
    avg_non_activation_time = np.mean(non_activation_durations) if non_activation_durations else 0

    # Resultados promedios
    averages = {
        'Average_Activation_Time': avg_activation_time,
        'Average_Non_Activation_Time': avg_non_activation_time
    }

    return stimulus_times_df, averages



def plot_emg_channels(database, mat_file, grasp_number, interactive=False, time=True, include_rest=False, 
                      padding=10, use_stimulus=False, addFourier=False, title=None):
    """
    Grafica los datos EMG de cada canal en subplots dentro de la misma figura.

    Parameters:
        database (str): Nombre de la base de datos.
        mat_file (dict): Archivo .mat cargado con datos EMG y estímulos.
        grasp_number (int): Número de agarre específico a graficar.
        interactive (bool): Activar modo interactivo de matplotlib.
        time (bool): Mostrar el tiempo en el eje x si es True, de lo contrario índices.
        include_rest (bool): Incluir periodo de reposo.
        padding (int): Tiempo de padding para agregar antes/después de los estímulos.
        use_stimulus (bool): Utilizar estímulos filtrados.
        addFourier (bool): Agregar el espectro de Fourier al final.
    """
    try:
        # Extraer datos EMG y estímulos
        emg_data, restimulus_data = db_utils.extract_data(mat_file, use_stimulus)
    except KeyError as e:
        print(f"KeyError in extract_data: {e}")
        raise

    # Obtener frecuencia
    frequency = DATABASE_INFO[database]['frequency'] if time else None
    
    if emg_data is None or restimulus_data is None:
        raise ValueError("EMG or Restimulus data is None")
    
    # Filtrar los datos
    try:
        filtered_emg_data, filtered_restimulus_data = db_utils.filter_data(
            emg_data, restimulus_data, grasp_number, include_rest, padding=padding)
    except KeyError as e:
        print(f"KeyError in filter_data: {e}")
        raise
    
    # Calcular eje x (tiempo o índices)
    num_samples = filtered_emg_data.shape[0]
    if time and frequency:
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(num_samples)
        x_label = "Samples"

    # Configurar modo interactivo
    if interactive:
        plt.ion()
    else:
        plt.ioff()

    # Crear figura y subplots para cada canal
    num_channels = filtered_emg_data.shape[1]  # Número de canales de EMG
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
    if title is None:
        fig.suptitle(f"EMG Data - Grasp {grasp_number}", fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)

    # Asegurar que axes sea iterable, incluso si es solo un subplot
    if num_channels == 1:
        axes = [axes]

    # Graficar cada canal
    for i, ax in enumerate(axes):
        ax.plot(x_axis, filtered_emg_data[:, i], label=f'Channel {i+1}', color='b')
        ax.set_ylabel(f'Channel {i+1}')
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel(x_label)  # Etiqueta en el último subplot

    # Mostrar la figura
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Fourier opcional
    if addFourier:
        plot_fourier_transform_with_envelope(filtered_emg_data, frequency)


def plot_single_emg_channel(database, raw_data, processed_data, grasp_number, channel, start=0, end=None, 
                            time=True, include_rest=False, padding=10, length=0.0, 
                            use_stimulus=False, addFourier=False):
    """
    Plots a given channel from the raw and processed EMG DataFrames.

    Parameters:
    - database: Name of the database.
    - raw_data: DataFrame containing the raw EMG signals.
    - processed_data: DataFrame containing the processed EMG signals.
    - grasp_number: The grasp number to filter.
    - channel: The channel to plot.
    - start: The starting index for the plot (default is 0).
    - end: The ending index for the plot (default is None, which means plot till the end).
    - time: Boolean indicating whether to use time on the x-axis.
    - include_rest: Boolean indicating whether to include rest periods.
    - padding: Padding to add before and after the stimulus.
    - length: Length of the test time to plot (in seconds).
    - use_stimulus: Boolean indicating whether to use filtered stimulus.
    - addFourier: Boolean indicating whether to add Fourier transform plots.
    """
    # Filter the raw and processed data
    filtered_raw_data = db_utils.filter_data_pandas(raw_data, grasp_number, include_rest=include_rest, padding=padding)
    filtered_processed_data = db_utils.filter_data_pandas(processed_data, grasp_number, include_rest=include_rest, padding=padding)
    
    if length > 0.01:
        final_time = filtered_raw_data['Time (s)'].iloc[0] + length
        filtered_raw_data = filtered_raw_data[filtered_raw_data['Time (s)'] < final_time]
        filtered_processed_data = filtered_processed_data[filtered_processed_data['Time (s)'] < final_time]

    # Extract the EMG channels
    raw_emg = prep_utils.extract_emg_channels(filtered_raw_data)
    processed_emg = prep_utils.extract_emg_channels(filtered_processed_data)

    # Get frequency if time is True
    if time:
        frequency = DATABASE_INFO[database]['frequency']
        num_samples = raw_emg.shape[0]
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(start, end)
        x_label = "Samples"

    plt.figure(figsize=(14, 6))

    # Plot the raw EMG signal
    plt.plot(x_axis, raw_emg[channel], label=f'Raw EMG - Channel {channel}', color='orange', alpha=0.7)

    # Plot the processed EMG signal
    plt.plot(x_axis, processed_emg[channel], label=f'Processed EMG - Channel {channel}', color='blue')

    plt.title(f'Raw and Processed EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()

    # Add transition marks
    if 'relabeled' in filtered_processed_data.columns:
        # Map transition indexes to the x_axis
        start_index, end_index = prep_utils.get_transition_indexes(filtered_processed_data['relabeled'].values)
        start_times = x_axis[start_index]  # Map start indexes to x_axis
        end_times = x_axis[end_index]      # Map end indexes to x_axis

        for time in start_times:
            plt.axvline(x=time, color='red', linestyle='--', linewidth=0.8, label='Start Transition')
        for time in end_times:
            plt.axvline(x=time, color='blue', linestyle='--', linewidth=0.8, label='End Transition')

    plt.tight_layout()
    plt.show()

    # Plot Fourier transform if requested
    if addFourier:
        plot_fourier_transform_with_envelope(processed_emg, frequency)


def plot_emg_channel_with_envelopes(database, raw_data, processed_data_list, grasp_number, channel, 
                                    start=0, end=None, time=True, include_rest=True, 
                                    padding=10, length=0.0, use_stimulus=False, addFourier=False):
    """
    Plots a given channel from the raw and processed EMG DataFrames, along with transformed envelopes.

    Parameters:
    - database: Name of the database.
    - raw_data: DataFrame containing the raw EMG signals.
    - processed_data_list: List of DataFrames containing the processed EMG signals.
    - grasp_number: The grasp number to filter.
    - channel: The channel to plot.
    - start: The starting index for the plot (default is 0).
    - end: The ending index for the plot (default is None, which means plot till the end).
    - time: Boolean indicating whether to use time on the x-axis.
    - include_rest: Boolean indicating whether to include rest periods.
    - padding: Padding to add before and after the stimulus.
    - length: Length of the test time to plot (in seconds).
    - use_stimulus: Boolean indicating whether to use filtered stimulus.
    - addFourier: Boolean indicating whether to add Fourier transform plots.
    """
    # Filter the raw data
    filtered_raw_data = db_utils.filter_data_pandas(raw_data, grasp_number, include_rest=include_rest, padding=padding)
    
    if length > 0.01:
        final_time = filtered_raw_data['Time (s)'].iloc[0] + length
        filtered_raw_data = filtered_raw_data[filtered_raw_data['Time (s)'] < final_time]

    # Extract the raw EMG channels
    raw_emg = prep_utils.extract_emg_channels(filtered_raw_data)

    # Get frequency if time is True
    if time:
        frequency = DATABASE_INFO[database]['frequency']
        num_samples = raw_emg.shape[0]
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(start, end)
        x_label = "Samples"

    plt.figure(figsize=(14, 6))

    # Plot the raw EMG signal
    plt.plot(x_axis, raw_emg[channel], label=f'Raw EMG - Channel {channel}', color='orange', alpha=0.7)

    # Iterate through the list of processed DataFrames and plot each one
    for i, processed_data in enumerate(processed_data_list):
        # Filter the processed data
        filtered_processed_data = db_utils.filter_data_pandas(processed_data, grasp_number, include_rest=include_rest, padding=padding)
        
        if length > 0.01:
            filtered_processed_data = filtered_processed_data[filtered_processed_data['Time (s)'] < final_time]

        # Extract the processed EMG channels
        processed_emg = prep_utils.extract_emg_channels(filtered_processed_data)

        # Plot the processed EMG signal
        plt.plot(x_axis, processed_emg[channel], label=f'Processed EMG {i+1} - Channel {channel}', alpha=0.7)

    plt.title(f'Raw and Processed EMG Signal with Envelopes - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()

    # Add transition marks (using the first processed DataFrame as reference)
    if 'relabeled' in filtered_processed_data.columns:
        start_index, end_index = prep_utils.get_transition_indexes(filtered_processed_data['relabeled'].values)
        start_times = x_axis[start_index]  # Map start indexes to x_axis
        end_times = x_axis[end_index]      # Map end indexes to x_axis

        for time in start_times:
            plt.axvline(x=time, color='red', linestyle='--', linewidth=0.8, label='Start Transition')
        for time in end_times:
            plt.axvline(x=time, color='blue', linestyle='--', linewidth=0.8, label='End Transition')

    plt.tight_layout()
    plt.show()

    # Plot Fourier transform if requested (using the first processed DataFrame as reference)
    if addFourier:
        plot_fourier_transform_with_envelope(processed_emg, frequency)

def plot_emg_windowed(database, mat_file, grasp_number, windowing, interactive=False, time=True, include_rest=False, padding=10, use_stimulus=False, addFourier=False, title=None):
    try:
        emg_data, restimulus_data = db_utils.extract_data(mat_file, use_stimulus)
    except KeyError as e:
        print(f"KeyError in extract_data: {e}")
        raise

    if time:
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
        filtered_emg_data, filtered_restimulus_data = db_utils.filter_data(
            emg_data, restimulus_data, grasp_number, include_rest, padding=padding)
    except KeyError as e:
        print(f"KeyError in filter_data: {e}")
        raise

    print(f"Filtered EMG data shape: {filtered_emg_data.shape}")
    print(f"Filtered restimulus data shape: {filtered_restimulus_data.shape}")

    if filtered_emg_data is None or filtered_restimulus_data is None:
        raise ValueError("Filtered data is None")

    # Definir la duración de la ventana en muestras
    window_size = int(windowing * frequency)  # 100 ms en muestras
    
    if window_size > filtered_emg_data.shape[0]:
        raise ValueError("Window size is larger than available data.")
    
    windowed_data = filtered_emg_data[:window_size, :]
    time_axis = np.arange(window_size) / frequency

    plt.figure(figsize=(10, 6))
    for i in range(windowed_data.shape[1]):
        plt.plot(time_axis, windowed_data[:, i], label=f'Channel {i+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('EMG Signal')
    plt.title(title if title else 'EMG Windowed Signal (100 ms)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_emg_data_basic(emg_data, frequency=2000, title=None):
    """
    Grafica las señales EMG contenidas en un DataFrame.
    
    Parámetros:
    - emg_data: DataFrame con señales EMG (columnas = canales).
    - frequency: Frecuencia de muestreo en Hz (predeterminado: 1000 Hz).
    - title: Título de la gráfica.
    """
    time_axis = [i / frequency for i in range(len(emg_data))] 

    plt.figure(figsize=(12, 6))
    
    for column in emg_data.columns:
        plt.plot(time_axis, emg_data[column], label=column)

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud EMG")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def plot_emg_channel_with_envelopes_fixed(database, raw_data, processed_data_list, grasp_number, channel, 
                                          start=0, end=None, time=True, include_rest=True, 
                                          padding=10, length=0.0, use_stimulus=False, addFourier=False):
    """
    Plots a given channel from the raw and processed EMG DataFrames, along with transformed envelopes.

    Parameters:
    - database: Name of the database.
    - raw_data: DataFrame containing the raw EMG signals.
    - processed_data_list: List of DataFrames containing the processed EMG signals.
    - grasp_number: The grasp number to filter.
    - channel: The channel to plot.
    - start: The starting index for the plot (default is 0).
    - end: The ending index for the plot (default is None, which means plot till the end).
    - time: Boolean indicating whether to use time on the x-axis.
    - include_rest: Boolean indicating whether to include rest periods.
    - padding: Padding to add before and after the stimulus.
    - length: Length of the test time to plot (in seconds).
    - use_stimulus: Boolean indicating whether to use filtered stimulus.
    - addFourier: Boolean indicating whether to add Fourier transform plots.
    """

    # Filtrar los datos en bruto
    filtered_raw_data = db_utils.filter_data_pandas(raw_data, grasp_number, include_rest=include_rest, padding=padding)

    # Verificar si los datos están vacíos o son None
    if filtered_raw_data is None or filtered_raw_data.empty:
        print(f"⚠️ Warning: No data found after filtering for grasp {grasp_number}. Skipping plot.")
        return

    # Extraer los canales EMG del raw
    raw_emg = prep_utils.extract_emg_channels(filtered_raw_data)

    # Determinar la escala del eje x
    if time:
        frequency = DATABASE_INFO[database]['frequency']
        num_samples = raw_emg.shape[0]
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(start, end) if end else np.arange(start, raw_emg.shape[0])
        x_label = "Samples"

    # Si length > 0, recortar los datos al tiempo especificado
    if length > 0.01:
        final_time = filtered_raw_data['Time (s)'].iloc[0] + length
        filtered_raw_data = filtered_raw_data[filtered_raw_data['Time (s)'] < final_time]

    # Crear la figura
    plt.figure(figsize=(14, 6))

    # Graficar la señal EMG cruda
    plt.plot(x_axis[:len(raw_emg[channel])], raw_emg[channel], label=f'Raw EMG - Channel {channel}', color='orange', alpha=0.7)

    # Iterar sobre los datos procesados y graficarlos
    for i, processed_data in enumerate(processed_data_list):
        # Filtrar los datos procesados
        filtered_processed_data = db_utils.filter_data_pandas(processed_data, grasp_number, include_rest=include_rest, padding=padding)
        
        if filtered_processed_data is None or filtered_processed_data.empty:
            print(f"⚠️ Warning: No processed data available for grasp {grasp_number}, skipping this entry.")
            continue
        
        # Recortar si length > 0
        if length > 0.01:
            filtered_processed_data = filtered_processed_data[filtered_processed_data['Time (s)'] < final_time]

        # Extraer los canales EMG procesados
        processed_emg = prep_utils.extract_emg_channels(filtered_processed_data)

        # Graficar la señal procesada
        plt.plot(x_axis[:len(processed_emg[channel])], processed_emg[channel], 
                 label=f'Processed EMG {i+1} - Channel {channel}', alpha=0.7)

    plt.title(f'Raw and Processed EMG Signal with Envelopes - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()

    # Agregar marcas de transición
    if 'relabeled' in filtered_raw_data.columns:
        start_index, end_index = prep_utils.get_transition_indexes(filtered_raw_data['relabeled'].values)
        start_times = x_axis[start_index] if len(start_index) > 0 else []
        end_times = x_axis[end_index] if len(end_index) > 0 else []

        for time in start_times:
            plt.axvline(x=time, color='red', linestyle='--', linewidth=0.8, label='Start Transition')
        for time in end_times:
            plt.axvline(x=time, color='blue', linestyle='--', linewidth=0.8, label='End Transition')

    plt.tight_layout()
    plt.show()

    # Graficar la transformada de Fourier si se solicita
    if addFourier:
        plot_fourier_transform_with_envelope(processed_emg, frequency)
