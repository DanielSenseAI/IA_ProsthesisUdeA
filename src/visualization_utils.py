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
from src.preprocessing_utils import get_transition_indexes
from src.preprocessing_utils import extract_emg_channels

def plot_data(filtered_emg_data, restimulus_data, grasp_number=None, interactive=False, frequency=None, title=None):
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


def plot_emg_dataframe(database, emg_data, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False, addFourier = False, length = 0.0): 
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


def plot_fourier_transform_with_envelope(emg_data, frequency, start_freq=0, end_freq=600, window_length=30, polyorder=6, print_max = True):
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
        max_freq = freqs[np.argmax(smoothed_magnitude[:, i])]
        if print_max:
            print(f"Max smoothed frequency for Channel {i + 1}: {max_freq} Hz")
    
    plt.title('Smoothed Envelope of Fourier Transform of EMG Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize=6)
    plt.xticks(np.arange(start_freq, end_freq + 1, step=10))  # Add more x-ticks
    plt.tight_layout()
    plt.show()


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


def plot_single_emg_channel(database, original_df, transformed_df, channel, start=5000, end=6000, time=True, addFourier=False):
    """
    Plots a given channel from the original and transformed EMG DataFrames.

    Parameters:
    - original_df: DataFrame containing the original EMG signals.
    - transformed_df: DataFrame containing the transformed EMG signals.
    - channel: The channel to plot.
    - start: The starting index for the plot (default is 0).
    - end: The ending index for the plot (default is None, which means plot till the end).
    - time: Boolean indicating whether to use time on the x-axis.
    - addFourier: Boolean indicating whether to add Fourier transform plots.
    """
    if end is None:
        end = len(original_df)

    # Extract the EMG channels for the specified indices
    initialEMG = extract_emg_channels(original_df.iloc[start:end])
    transformedEMG = extract_emg_channels(transformed_df.iloc[start:end])

    # Get frequency if time is True
    if time:
        frequency = DATABASE_INFO[database]['frequency']
        num_samples = initialEMG.shape[0]
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(start, end)
        x_label = "Samples"

    plt.figure(figsize=(14, 9))

    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.plot(x_axis, initialEMG[channel])
    plt.title(f'Original EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')

    # Plot transformed signal
    plt.subplot(3, 1, 2)
    plt.plot(x_axis, transformedEMG[channel])
    plt.title(f'Transformed EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')

    # Plot overlaid signals
    plt.subplot(3, 1, 3)
    plt.plot(x_axis, initialEMG[channel], label='Original')
    plt.plot(x_axis, transformedEMG[channel], label='Transformed', alpha=0.7)
    plt.title(f'Overlay of Original and Transformed EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot Fourier transform if requested
    if addFourier:
        plot_fourier_transform_with_envelope(transformedEMG, frequency)


def plot_emg_channel_with_envelopes(database, original_df, transformed_dfs, channel, start=5000, end=6000, time=True, addFourier=False):
    """
    Plots a given channel from the original and multiple transformed EMG DataFrames.

    Parameters:
    - original_df: DataFrame containing the original EMG signals.
    - transformed_dfs: List of DataFrames containing the transformed EMG signals.
    - channel: The channel to plot.
    - start: The starting index for the plot (default is 0).
    - end: The ending index for the plot (default is None, which means plot till the end).
    - time: Boolean indicating whether to use time on the x-axis.
    - addFourier: Boolean indicating whether to add Fourier transform plots.
    """
    if end is None:
        end = len(original_df)

    # Extract the EMG channels for the specified indices
    initialEMG = extract_emg_channels(original_df.iloc[start:end])

    # Get frequency if time is True
    if time:
        frequency = DATABASE_INFO[database]['frequency']
        num_samples = initialEMG.shape[0]
        x_axis = np.linspace(0, num_samples / frequency, num_samples)
        x_label = "Time (s)"
    else:
        x_axis = np.arange(start, end)
        x_label = "Samples"

    plt.figure(figsize=(14, 9))

    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, initialEMG[channel])
    plt.title(f'Original EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')

    # Plot overlaid signals
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, initialEMG[channel], label='Original')
    for i, transformed_df in enumerate(transformed_dfs):
        transformedEMG = extract_emg_channels(transformed_df.iloc[start:end])
        plt.plot(x_axis, transformedEMG[channel], label=f'Transformed (Envelope Type {i+1})', alpha=0.7)
    plt.title(f'Overlay of Original and Transformed EMG Signal - Channel {channel}')
    plt.xlabel(x_label)
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot Fourier transform if requested
    if addFourier:
        plot_fourier_transform_with_envelope(initialEMG, frequency)
