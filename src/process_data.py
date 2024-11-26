import numpy as np
from scipy.stats import kurtosis, skew


def calculate_rms(signal: np.array) -> float:
    """
    Calcula el valor cuadrático medio (RMS, Root Mean Square) de una señal dada.

    Parámetros:
    ----------
    signal : np.array
        Un array que representa la señal de la cual se calculará el valor cuadrático medio. Puede ser una señal EMG, de audio, o cualquier otro tipo de datos numéricos.

    Retorna:
    -------
    float
        El valor cuadrático medio (RMS) de la señal.
    """
    squared_signal = np.square(signal)
    mean_squared = np.mean(squared_signal)
    rms = np.sqrt(mean_squared)
    return rms


def calculate_mav(signal: np.array) -> float:
    """
        Calcula la media del valor absoluto (MAV, Mean Absolute Value) de una señal.

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal de la cual se calculará la MAV.

        Retorna:
        -------
        float
            La media del valor absoluto (MAV) de la señal.
        
    """
    absolute_values = [abs(x) for x in signal]
    mav = sum(absolute_values) / len(signal)
    return mav


def calculate_mavs(signal: np.array) -> float:
    """
    Calcula la media de los valores absolutos de las diferencias sucesivas (MAVS) de una señal.

    Parámetros:
    ----------
    signal : np.array
        Un array que representa la señal de la cual se calculará la MAVS.

    Retorna:
    -------
    float
        La media del valor absoluto de las diferencias sucesivas (MAVS) de la señal.

    """
    absolute_diffs = [abs(signal[i] - signal[i-1]) for i in range(1, len(signal))]
    mavs = sum(absolute_diffs) / (len(signal) - 1)
    return mavs


def calculate_variance(signal: np.array) -> float:
    """
        Calcula la varianza de una señal.

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal de la cual se calculará la varianza.

        Retorna:
        -------
        float
            La varianza de la señal.
    """
    variance = np.var(signal)
    return variance


def calculate_sample_variance(signal: np.array) -> float:
    """
        Calcula la varianza muestral de una señal (utilizando Bessel's correction).

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal de la cual se calculará la varianza muestral.

        Retorna:
        -------
        float
            La varianza muestral de la señal.
    """
    sample_variance = np.var(signal, ddof=1)
    return sample_variance


def calculate_kurtosis(signal: np.array) -> float:
    """
        Calcula la curtosis de una señal.

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal de la cual se calculará la curtosis.

        Retorna:
        -------
        float
            El valor de la curtosis de la señal.
    """
    kurtosis_value = kurtosis(signal)
    return kurtosis_value


def calculate_skewness(signal: np.array) -> float:
    """
        Calcula la asimetría (skewness) de una señal.

        Parámetros:
        ----------
        signal : np.array
            Un array que representa la señal de la cual se calculará la asimetría.

        Retorna:
        -------
        float
            El valor de la asimetría de la señal.
    """
    skewness_symetric = skew(signal)
    return skewness_symetric