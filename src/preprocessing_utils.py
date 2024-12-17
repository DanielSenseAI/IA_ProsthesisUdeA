from collections import Counter
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from typing import Dict


def get_stimulus_index(stimulus):
    """
    Obtiene los índices de inicio y fin de un estímulo en una señal binaria.

    Esta función toma como entrada un arreglo o lista de estímulos binarios, 
    donde los valores distintos de cero indican la presencia de un estímulo 
    (es decir, 1) y los ceros indican ausencia del estímulo (es decir, 0). 
    El objetivo es identificar los momentos en que un estímulo comienza y 
    termina.

    Parámetros:
    -----------
    stimulus : np.array o lista
        Un array o lista unidimensional que representa la señal del estímulo, 
        donde los valores distintos de cero indican la presencia del estímulo.

    Retorna:
    --------
    start_index : np.array
        Un array que contiene los índices donde comienza cada estímulo. 
        Es decir, los índices donde el valor cambia de 0 a un valor distinto 
        de cero.
        
    end_index : np.array
        Un array que contiene los índices donde termina cada estímulo. 
        Es decir, los índices donde el valor cambia de un valor distinto de 
        cero a 0.
    """
    start_index = np.nonzero((stimulus[1:] != 0) & (stimulus[:-1] == 0))[0]
    end_index = np.nonzero((stimulus[1:] == 0) & (stimulus[:-1] != 0))[0]
    return start_index, end_index


def get_start_end_index(index_array_start: np.array, index_array_end: np.array, movements: np.array, repetition: int)-> dict:
    """
    Filtra y organiza los índices de inicio y fin de movimientos en una serie de estímulos.

    Esta función toma los arrays de índices de inicio y fin, junto con una lista 
    de movimientos y un número de repeticiones, para obtener subconjuntos de índices 
    correspondientes a cada movimiento. Devuelve un diccionario con los índices de inicio 
    y fin de cada movimiento filtrado.

    Parámetros:
    -----------
    index_array_start : np.array
        Un array que contiene los índices donde comienzan los estímulos para todos los movimientos.
        
    index_array_end : np.array
        Un array que contiene los índices donde terminan los estímulos para todos los movimientos.
        
    movements : np.array
        Un array con los identificadores de los movimientos que se quieren filtrar. 
        Los valores deben ser enteros positivos, ya que se ajustan con base 1 (1 corresponde al primer movimiento).

    repetition : int
        Número de repeticiones a considerar para cada movimiento, determinando cuántos índices de inicio y fin
        se toman a partir del valor base de cada movimiento.

    Retorna:
    --------
    filtered_index : dict
        Un diccionario donde las claves son los movimientos (convertidos a cadena de texto), 
        y los valores son diccionarios con dos entradas:
        - 'start': Un array con los índices de inicio de ese movimiento.
        - 'end': Un array con los índices de fin de ese movimiento.
    """
    filtered_index = {}
    for i in movements:
        i=i-1
        index_start = index_array_start[i*10:(i*10)+repetition]
        if i*10 == 0:
            index_end = index_array_end[0:(i*10)+repetition]
            index_end = np.insert(index_end, 0, 0)
        else: 
            index_end = index_array_end[(i*10)-1:(i*10)+repetition]
        data = {
            'start': index_start,
            'end': index_end
        }
        filtered_index[str(i+1)] = data

    return filtered_index


def get_signal_by_movement_complete(signal: np.array, movement: dict) -> np.array:
    """
        Extrae una señal segmentada de un array de señales basado en la información de un diccionario que contiene los puntos de inicio y fin de movimientos.

        Parámetros:
        ----------
        signal : np.array
            Un array de señales del cual se extraerá un segmento. Puede representar datos como lecturas de un sensor, señales temporales, etc.
        
        movement : dict
            Un diccionario que contiene información sobre los puntos de inicio y fin del movimiento. Debe tener la siguiente estructura:
            - 'end' : list[int]
                Una lista de índices que representan los puntos finales de diferentes movimientos. Se utilizarán para segmentar la señal.

        Retorna:
        -------
        np.array
            Un segmento de la señal original, que corresponde a los índices entre el primer y el último valor en la lista `movement['end']`.
    """
    segmented_signal = signal[(movement['end'][0])+1:movement['end'][-1]]
    return segmented_signal

def get_transition_indexes(data):
    # Check if the input is a DataFrame
    if isinstance(data, pd.DataFrame):
        # Get the values from the DataFrame and flatten them
        values = data.values.flatten()
    elif isinstance(data, np.ndarray):
        # Flatten the numpy array to ensure it is one-dimensional
        values = data.flatten()
    else:
        raise TypeError("Input must be a pandas DataFrame or a numpy array")
     
    # Find indexes where the value changes from 0 to non-zero
    zero_to_nonzero = np.where((values[:-1] == 0) & (values[1:] != 0))[0] + 1
    
    # Find indexes where the value changes from non-zero to 0
    nonzero_to_zero = np.where((values[:-1] != 0) & (values[1:] == 0))[0] + 1

    print(f'Number of Repetitions: {len(zero_to_nonzero)}')
    
    return zero_to_nonzero, nonzero_to_zero


def get_envelope(emg_signal: np.array) -> np.array:
    """
        Calcula la envolvente de una señal EMG (Electromiografía) utilizando la Transformada de Hilbert.

        Parámetros:
        ----------
        emg_signal : np.array
            Un array que representa la señal EMG (Electromiografía). La señal puede estar en forma cruda o preprocesada.

        Retorna:
        -------
        np.array
            La envolvente de la señal EMG. Es un array de la misma longitud que `emg_signal`, que contiene los valores de amplitud instantánea de la señal.

        Ejemplo:
        --------
        emg_signal = np.array([0.1, 0.5, -0.3, 0.8, -0.2])
        envelope = get_envelope(emg_signal)
        print(envelope)  # Salida: np.array([...]) (La envolvente de la señal EMG)
    """
    envelope = np.abs(hilbert(emg_signal))
    return envelope


def get_filtered_signal(signal: np.array, fc: float, fm: float) -> np.array:
    """
        Aplica un filtro pasabajo (low-pass) a una señal utilizando un filtro Butterworth.

        Parámetros:
        ----------
        signal : np.array
            El array que representa la señal que se desea filtrar.
        
        fc : float
            La frecuencia de corte del filtro pasabajo, en Hz. Las frecuencias por encima de esta serán atenuadas.
        
        fm : float
            La frecuencia de muestreo de la señal, en Hz. Es necesaria para calcular la frecuencia de Nyquist.

        Retorna:
        -------
        np.array
            La señal filtrada, después de aplicar el filtro pasabajo de Butterworth.
    """
    nyquist = 0.5 * fm
    normal_frec_corte = fc / nyquist
    b, a = butter(4, normal_frec_corte, btype='low', analog=False)
    return filtfilt(b, a, signal)


def get_envelope_filtered(emg_signal: np.array, fc:float, fm:float) ->np.array:
    """
        Aplica un filtro pasabajo a una señal EMG y luego calcula la envolvente de la señal filtrada.

        Parámetros:
        ----------
        emg_signal : np.array
            Un array que representa la señal de Electromiografía (EMG).
        
        fc : float
            La frecuencia de corte del filtro pasabajo en Hz. Las frecuencias por encima de este valor serán atenuadas.
        
        fm : float
            La frecuencia de muestreo de la señal EMG en Hz. Es necesaria para calcular la frecuencia de Nyquist.

        Retorna:
        -------
        np.array
            La envolvente de la señal EMG filtrada, que representa los valores de amplitud instantánea de la señal procesada.
    """
    filtered_signal = get_filtered_signal(emg_signal, fc, fm)
    envelope = get_envelope(filtered_signal)
    return envelope


def create_windows_with_overlap(signal, window_length,  overlap):
    """
        Divide una señal en ventanas de longitud fija, permitiendo superposición entre ventanas.

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal que se va a dividir en ventanas.

        window_length : int
            La longitud de cada ventana (cantidad de muestras por ventana).

        overlap : float
            El porcentaje de superposición entre ventanas. Si es 0, no habrá superposición.

        Retorna:
        -------
        list
            Una lista de ventanas (subarrays) extraídas de la señal original, cada una con una longitud de `window_length`.
    """
    if overlap == 0:
        return [signal[i:i+window_length] for i in range(0, len(signal), window_length)]
    else:
        overlap_size = int(window_length * overlap / 100)
        step_size = window_length - overlap_size
        return [signal[i:i+window_length] for i in range(0, len(signal) - window_length + 1, step_size)]
    

def get_label(signal, percentage, labels):
    """
        Asigna una etiqueta a una señal basada en la frecuencia de sus valores.

        La función cuenta la frecuencia de los elementos en la señal y compara la frecuencia 
        de cada valor con un umbral determinado por el porcentaje especificado. Si la frecuencia 
        de un valor excede el umbral, se devuelve la etiqueta correspondiente a ese valor.

        Parámetros:
        ----------
        signal : list
            Una lista de listas (por ejemplo, ventanas de una señal) que contiene los valores 
            de la señal.

        percentage : float
            El porcentaje que se usará para calcular el umbral de frecuencia. Si un valor 
            tiene una frecuencia igual o mayor que este umbral, se le asignará la etiqueta.

        labels : dict
            Un diccionario que mapea valores a etiquetas. Las claves deben ser cadenas 
            que representan los valores de la señal.

        Retorna:
        -------
        str
            La etiqueta asociada al valor que cumple con el umbral de frecuencia. 
            Si ningún valor cumple con el umbral, se retorna 'None'.
    """
    flattened_signal = [item for sublist in signal for item in sublist]
    
    counter = Counter(flattened_signal)
    total_elements = len(signal)
    threshold = (percentage / 100) * total_elements
    for value, frequency in counter.items():
        # if value != 0 and frequency >= threshold:
        if frequency >= threshold:
            label = labels[str(value)]
            return label
    return 'None' # If no value meets the threshold    return 'None' # If no value meets the threshold

def add_time(emg_data, frequency):
    if isinstance(emg_data, pd.DataFrame):
        time = [i / frequency for i in range(len(emg_data))]
        emg_data['Time (s)'] = time
    else:
        raise TypeError("emg_data must be a pandas DataFrame")
    return emg_data

def relabel_database(database, stimulus, exercise = None):
    """
    Adds a relabeled column to the DataFrame based on the exercise type and the movements library.

    Parameters:
        database (pd.DataFrame): The DataFrame containing the stimulus data.
        movements_library (dict): Dictionary mapping numeric labels to movement names.
        stimulus_column (str): The name of the column in the DataFrame containing numeric labels.
        exercise (str): The exercise type ('A', 'B', 'C', 'D') to adjust the label mapping.

    Returns:
        pd.DataFrame: The updated DataFrame with an additional 'Relabeled' column.
    """
    db_145_relation = ['A', 'B', 'C']
    db_23_relation = ['B', 'C', 'D']

    # Define label mappings for each exercise in DB1, 4 and 5
    exercise_label_mappings = {
        'A': {0: 0, 1: 50, 2: 51, 3: 52, 4: 53, 5: 54, 6: 55, 7: 56, 8: 57, 9: 58, 10: 59, 11: 60, 12: 61},
        'B': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17},
        'C': {0: 0, 1: 18, 2: 19, 3: 20, 4: 21, 5: 22, 6: 23, 7: 24, 8: 25, 9: 26, 10: 27, 11: 28, 12: 29, 13: 30, 14: 31, 15: 32, 16: 33, 17: 34, 18: 35, 19: 36, 20: 37, 21: 38, 22: 39, 23: 40},
        'D': {0: 0, 1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 46, 7: 47, 8: 48, 9: 49}
    }

   # Define the custom mapping for DB8
    DB8_mapping = {0: 0, 1: 58, 2: 60, 3: 50, 4: 52, 5: 3, 6: 7, 7: 18, 8: 34, 9: 30}

    # Perform relabeling
    if database in ['DB1', 'DB4', 'DB5']:
        exercise_label = db_145_relation[exercise-1]
        print(f'new exercise label: {exercise_label}')
        label_mapping = exercise_label_mappings[exercise_label]
        stimulus['relabeled'] = stimulus['stimulus'].map(label_mapping)
        print(f'Relabeling performed for exercise {exercise} of {database}.')
    elif database == 'DB8':
        stimulus['relabeled'] = stimulus['stimulus'].map(DB8_mapping)
        print(f'Relabeling performed for DB8.')
    else:
        stimulus['relabeled'] = stimulus['stimulus']
        print("Warning: No relabeling performed! 'Relabeled' column contains original values.")

    # Check for missing labels
    missing_labels = 0
    missing_labels = stimulus['relabeled'].isna().unique()
    if missing_labels.size > 0:
        print(f"Warning: The following labels were not found in the mapping for exercise {exercise}: {missing_labels}")
    
    return stimulus

def extract_emg_channels(emg_df):
    emg_channel_columns = [col for col in emg_df.columns if col.startswith('Channel')]
    emg_channels_df = emg_df[emg_channel_columns]
    return emg_channels_df   



