#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Window Feature Extractor

This script processes EMG data from .mat files in a specified database:
1. Iterates over all .mat files in a database folder
2. Opens each file using the src functions
3. Builds a dataframe from the data
4. Splits the dataframe into windows
5. Calculates features for each window
6. Builds a dataframe including all features
7. Adds the label from the "relabeled" column 
8. Saves results to a parquet file (appending if not the first file)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Use non-interactive backend for saving plots
from scipy.io import loadmat, whosmat
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import traceback
from multiprocessing import Pool, cpu_count

# Import src modules
import src
from src import loadmatNina
from src.preprocessing_utils import get_envelope


# Add at the beginning of your file, after imports for CUDA SUPPORT
import time as time_module  # Rename to avoid conflict
import os
import sys
import subprocess
import traceback

# Function to get environment information
def get_env_info():
    info = {
        "os_name": os.name,
        "platform": sys.platform,
        "python_version": sys.version,
        "path": os.environ.get("PATH", ""),
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "cuda_home": os.environ.get("CUDA_HOME", "")
    }
    return info

# Function to check CUDA with system commands
def check_cuda_system():
    results = {}
    
    # Check for nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        results["nvidia-smi"] = "Available" if result.returncode == 0 else "Failed"
        if result.returncode == 0:
            # Extract driver version
            for line in result.stdout.split('\n'):
                if "Driver Version:" in line:
                    results["driver_version"] = line.split("Driver Version:")[1].strip().split()[0]
                    break
    except:
        results["nvidia-smi"] = "Not available"
    
    # Check for nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        results["nvcc"] = "Available" if result.returncode == 0 else "Failed"
        if result.returncode == 0:
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if "release" in line and "V" in line:
                    parts = line.split("V")
                    if len(parts) > 1:
                        results["cuda_version"] = parts[1].strip().split()[0]
                        break
    except:
        results["nvcc"] = "Not available"
    
    return results

# Initialize CUDA flag
USE_GPU = False
DEVICE = None

def setup_cuda(force_cpu=False):
    """
    Set up CUDA environment and check availability
    Returns True if CUDA is available and working
    """
    global USE_GPU, DEVICE
    
    # If force_cpu, skip all checks
    if force_cpu:
        print("Forcing CPU usage as requested")
        return False
    
    print("\n=== CUDA Environment Check ===")
    env_info = get_env_info()
    print(f"OS: {env_info['platform']}")
    print(f"Python: {env_info['python_version']}")
    print(f"CUDA_PATH: {env_info['cuda_path']}")
    print(f"CUDA_HOME: {env_info['cuda_home']}")
    
    # Check system CUDA installation
    print("\n=== System CUDA Check ===")
    sys_cuda = check_cuda_system()
    print(f"NVIDIA Driver: {sys_cuda.get('nvidia-smi', 'Not found')}")
    if 'driver_version' in sys_cuda:
        print(f"Driver Version: {sys_cuda['driver_version']}")
    print(f"NVCC: {sys_cuda.get('nvcc', 'Not found')}")
    if 'cuda_version' in sys_cuda:
        print(f"CUDA Version: {sys_cuda['cuda_version']}")
    
    # Try importing torch and check CUDA
    print("\n=== PyTorch CUDA Check ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        
        if cuda_available:
            # Get device count and names
            device_count = torch.cuda.device_count()
            print(f"GPU count: {device_count}")
            
            # Print details of each device
            for i in range(device_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Get CUDA version
            if hasattr(torch.version, 'cuda'):
                print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            
            # Check cuDNN
            if hasattr(torch.backends, 'cudnn'):
                print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
                if torch.backends.cudnn.enabled:
                    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            
            # Test with tensor operations
            print("\nRunning basic CUDA tensor test...")
            try:
                # Create CUDA tensors and perform operations
                x = torch.tensor([1.0, 2.0], device="cuda")
                y = torch.tensor([3.0, 4.0], device="cuda")
                z = x + y
                print(f"Tensor calculation successful: {x} + {y} = {z}")
                
                # Set global variables
                USE_GPU = True
                DEVICE = torch.device("cuda")
                print("\n✓ CUDA is available and working")
                return True
                
            except Exception as e:
                print(f"❌ CUDA tensor operations failed: {e}")
                traceback.print_exc()
                print("\nCUDA appears to be installed but not functioning correctly.")
        else:
            print("❌ CUDA is not available according to PyTorch")
    
    except ImportError:
        print("❌ PyTorch is not installed")
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        traceback.print_exc()
    
    print("\nFalling back to CPU processing")
    return False


# Suppress output context manager
import sys
import io
import contextlib

# Add this context manager to your imports
@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress all stdout/stderr output."""
    # Save the current stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # Redirect stdout/stderr to a dummy object
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        # Return control to the caller
        yield
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Add right after your imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       module="pywt._multilevel")

# Try importing GPU libraries
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        DEVICE = torch.device("cuda")
        #print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        #print(f"Using GPU acceleration for feature extraction")
    else:
        print("CUDA-capable GPU not available. Using CPU only.")
except ImportError:
    print("PyTorch not installed. Using CPU only.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract features from windowed EMG data')
    
    parser.add_argument('--database', type=str, default='DB4',
                        help='Database name to process (default: DB4)')
    parser.add_argument('--window-size', type=int, default=200,
                        help='Window size in samples (default: 200, which is 100ms at 2kHz)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Overlap between windows in samples (default: 0)')
    parser.add_argument('--envelope-type', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        help='Envelope type to use (0=None, 1=Hilbert, 2=RMS, 3=MAV)')
    parser.add_argument('--filter-cutoff', type=float, default=1,
                        help='Low-pass filter cutoff frequency in Hz (default: 1.0)')
    parser.add_argument('--only-channel-10', action='store_true',
                        help='Process only Channel 10 (default: process all channels)')
    parser.add_argument('--filtered-labels', type=str, default='',
                        help='Comma-separated list of labels to include (default: empty, process all labels)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available (requires PyTorch)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for GPU processing (default: 32)')
    parser.add_argument('--fast-mode', action='store_true',
                   help='Skip expensive computations for faster processing')

    return parser.parse_args()


def frequency_domain_features(window, fs=2000):
    """Calculate frequency domain features of the EMG signal."""
    # Compute power spectrum
    freqs = np.fft.rfftfreq(len(window), d=1/fs)
    fft_values = np.fft.rfft(window)
    power_spectrum = np.abs(fft_values) ** 2
    
    # Ignore DC component
    if len(freqs) > 1:
        freqs = freqs[1:]
        power_spectrum = power_spectrum[1:]
    
    if len(power_spectrum) == 0 or np.sum(power_spectrum) == 0:
        return {
            "MDF": 0,      # Median frequency
            "PKF": 0,      # Peak frequency
            "MNP": 0,      # Mean power
            "TTP": 0,      # Total power
            "SM1": 0,      # First spectral moment
            "SM2": 0,      # Second spectral moment
            "SM3": 0,      # Third spectral moment
            "MNPF": 0      # Mean power frequency
        }
    
    # Calculate features
    total_power = np.sum(power_spectrum)
    
    # Median frequency - frequency that divides the power spectrum into two equal parts
    cumulative_power = np.cumsum(power_spectrum)
    median_idx = np.argmax(cumulative_power >= total_power/2)
    median_frequency = freqs[median_idx] if median_idx < len(freqs) else 0
    
    # Peak frequency - frequency with maximum power
    peak_idx = np.argmax(power_spectrum)
    peak_frequency = freqs[peak_idx] if len(freqs) > peak_idx else 0
    
    # Mean power
    mean_power = np.mean(power_spectrum)
    
    # Mean power frequency (weighted average of frequencies)
    mean_power_freq = np.sum(freqs * power_spectrum) / total_power if total_power > 0 else 0
    
    # Spectral moments
    sm1 = np.sum(freqs * power_spectrum) / total_power if total_power > 0 else 0
    sm2 = np.sum((freqs - sm1)**2 * power_spectrum) / total_power if total_power > 0 else 0
    sm3 = np.sum((freqs - sm1)**3 * power_spectrum) / total_power if total_power > 0 else 0
    
    return {
        "MDF": median_frequency,  # Median frequency
        "PKF": peak_frequency,    # Peak frequency
        "MNP": mean_power,        # Mean power
        "TTP": total_power,       # Total power
        "SM1": sm1,               # First spectral moment
        "SM2": sm2,               # Second spectral moment
        "SM3": sm3,               # Third spectral moment
        "MNPF": mean_power_freq   # Mean power frequency
    }


def complexity_features(window, fast_mode=True):
    if fast_mode:
        # Return simplified estimates
        return {
            "SampEn": np.std(window) / (np.mean(np.abs(window)) + 1e-10),  # Simple approximation
            "CC": np.mean(np.abs(np.diff(window))),
            "LE": np.log(np.max(np.abs(np.diff(window))) + 1e-10),
            "HFD": 0
        }
    
    """Calculate signal complexity features of the EMG signal."""
    n = len(window)
    if n <= 1:
        return {
            "SampEn": 0,
            "CC": 0,  # Correlation dimension
            "LE": 0,  # Largest Lyapunov exponent estimate
            "HFD": 0  # Higuchi fractal dimension
        }
        
    # Sample Entropy (simplified version)
    # For a more accurate version, consider using libraries like EntroPy or PyEEG
    std_dev = np.std(window)
    if std_dev == 0:
        return {
            "SampEn": 0,
            "CC": 0,
            "LE": 0,
            "HFD": 0
        }
        
    r = 0.2 * std_dev  # Typical value is 0.2 * std
    m = 2  # Embedding dimension
    
    # Count similar patterns
    def count_matches(template, data, r):
        return np.sum(np.abs(data - template) < r)
    
    # Calculate sample entropy
    try:
        count_m = 0
        count_m1 = 0
        
        for i in range(n - m):
            template_m = window[i:i+m]
            template_m1 = window[i:i+m+1]
            
            # Count matches for m and m+1 length templates
            for j in range(n - m):
                if i == j:
                    continue
                if count_matches(template_m, window[j:j+m], r) == m:
                    count_m += 1
                    if j < n - m - 1 and count_matches(template_m1, window[j:j+m+1], r) == m+1:
                        count_m1 += 1
        
        # Compute sample entropy
        samp_en = -np.log((count_m1 + 0.000001) / (count_m + 0.000001))
    except:
        samp_en = 0
    
    # Higuchi Fractal Dimension (simplified)
    try:
        kmax = 8  # Maximum delay
        L = np.zeros(kmax)
        x = np.array(window)
        N = len(x)
        
        for k in range(1, kmax + 1):
            Lk = 0
            for m in range(0, k):
                # Construct the sub-series
                indices = np.arange(m, N, k)
                Lmk = np.sum(np.abs(np.diff(x[indices]))) * (N - 1) / (((N - m) // k) * k)
                Lk += Lmk
            L[k-1] = Lk / k
        
        # Fit the curve to estimate fractal dimension
        x_values = np.log(np.arange(1, kmax + 1))
        y_values = np.log(L)
        hfd = -np.polyfit(x_values, y_values, 1)[0]
    except:
        hfd = 0
        
    # Simplified estimates for the other complexity measures
    # For accurate values, specialized algorithms should be used
    cc = np.mean(np.abs(np.diff(window)))  # Correlation dimension estimate
    le = np.log(np.max(np.abs(np.diff(window))) + 1e-10)  # Lyapunov exponent estimate
    
    return {
        "SampEn": samp_en,  # Sample entropy
        "CC": cc,           # Correlation dimension approximation
        "LE": le,           # Largest Lyapunov exponent approximation
        "HFD": hfd          # Higuchi fractal dimension
    }


def time_domain_advanced(window):
    """Calculate advanced time-domain features of the EMG signal."""
    if len(window) <= 1:
        return {
            "DASDV": 0,   # Difference absolute standard deviation value
            "MYOP": 0,    # Myopulse percentage rate
            "WAMP": 0,    # Willison amplitude
            "CARD": 0,    # Cardinality
            "LOG": 0,     # Log detector
            "SKEW": 0,    # Skewness
            "KURT": 0     # Kurtosis (added to main features already)
        }
    
    # Difference absolute standard deviation value
    dasdv = np.sqrt(np.mean(np.diff(window) ** 2))
    
    # Myopulse percentage rate (using 0.016 as threshold, about 1.6% of max amplitude)
    threshold = 0.016  # This should be adjusted based on your signal characteristics
    myop = np.sum(np.abs(window) > threshold) / len(window)
    
    # Willison amplitude (count of times signal changes by more than threshold)
    wamp_threshold = 0.01  # Adjust based on your signal
    wamp = np.sum(np.abs(np.diff(window)) > wamp_threshold)
    
    # Cardinality (approximation - number of unique values, normalized)
    # Using bins to approximate cardinality for continuous values
    hist, _ = np.histogram(window, bins=min(50, len(window)))
    card = np.sum(hist > 0) / min(50, len(window))
    
    # Log detector
    log_values = np.log(np.abs(window) + 1e-10)  # Adding small value to avoid log(0)
    log_detector = np.exp(np.mean(log_values))
    
    # Skewness
    skewness = 0
    std_dev = np.std(window)
    if std_dev > 0:
        mean_val = np.mean(window)
        skewness = np.mean(((window - mean_val) / std_dev) ** 3)
    
    # Kurtosis
    kurtosis = 0
    if std_dev > 0:
        mean_val = np.mean(window)
        kurtosis = np.mean(((window - mean_val) / std_dev) ** 4)
    
    return {
        "DASDV": dasdv,
        "MYOP": myop,
        "WAMP": wamp,
        "CARD": card,
        "LOG": log_detector,
        "SKEW": skewness,
        "KURT": kurtosis
    }


def calculate_features(window, fs=2000):
    """
    Calculate features for a window of EMG data.
    Uses GPU acceleration when available.
    
    Args:
        window: NumPy array containing the EMG data window
        fs: Sampling frequency in Hz (default: 2000)
        
    Returns:
        Dictionary of calculated features
    """
    try:
        # Convert to GPU if available and enabled
        if USE_GPU:
            # Convert to PyTorch tensor on GPU
            window_tensor = torch.tensor(window, dtype=torch.float32, device=DEVICE)
            abs_signal = torch.abs(window_tensor)
            diff_signal = window_tensor[1:] - window_tensor[:-1]
            diff_abs_signal = torch.abs(diff_signal)
            
            # Calculate basic metrics on GPU
            features = {
                "MAV": torch.mean(abs_signal).item(),
                "MAV_STD": torch.std(abs_signal).item(),
                "IAV": torch.sum(abs_signal).item(),
                "IAV_STD": torch.std(abs_signal).item(),
                "RMS": torch.sqrt(torch.mean(window_tensor**2)).item(),
                "RMS_STD": torch.std(window_tensor).item(),
                "WL": torch.sum(diff_abs_signal).item(),
                "WL_STD": torch.std(diff_abs_signal).item(),
                "VAR": torch.var(window_tensor).item(),
                "VAR_STD": torch.std(window_tensor).item(),
                "TD": torch.sum(diff_abs_signal).item(),
                "TD_STD": torch.std(diff_abs_signal).item(),
                "MAVS": torch.mean(diff_abs_signal).item(),
                "MAVS_STD": torch.std(diff_abs_signal).item(),
                "MNP": torch.mean(window_tensor**2).item(),
                "MNP_STD": torch.std(window_tensor**2).item(),
            }
            
            # Calculate zero crossings
            zero_crossings = (window_tensor[:-1] * window_tensor[1:]) < 0
            features["ZC"] = torch.sum(zero_crossings).item()
            features["ZC_STD"] = torch.std(zero_crossings.float()).item()
            
            # Calculate slope sign changes on GPU
            ssc_values = (diff_signal[1:] * diff_signal[:-1]) < 0
            features["SSC"] = torch.sum(ssc_values).item()
            features["SSC_STD"] = torch.std(ssc_values.float()).item()
            
            # Calculate coefficient of variation
            mean_signal = torch.mean(window_tensor).item()
            features["CoV"] = (torch.std(window_tensor).item() / mean_signal) if mean_signal != 0 else 0
            
            # Kurtosis on GPU
            if torch.std(window_tensor) != 0:
                normalized = (window_tensor - torch.mean(window_tensor)) / torch.std(window_tensor)
                features["Kurt"] = torch.mean(normalized**4).item()
            else:
                features["Kurt"] = 0
            features["Kurt_STD"] = 0  # Not meaningful for scalar Kurt
            
            # FFT on GPU
            window_np = window  # Some operations still need to be on CPU
            
        else:
            # Calculate basic time-domain features (CPU version)
            abs_signal = np.abs(window)
            diff_signal = np.diff(window)
            diff_abs_signal = np.abs(diff_signal)
            
            # Compute metrics
            features = {
                "MAV": np.mean(abs_signal),
                "MAV_STD": np.std(abs_signal),
                "IAV": np.sum(abs_signal),
                "IAV_STD": np.std(abs_signal),
                "RMS": np.sqrt(np.mean(window**2)),
                "RMS_STD": np.std(window),
                "WL": np.sum(diff_abs_signal),
                "WL_STD": np.std(diff_abs_signal),
                "ZC": np.sum(np.diff(np.sign(window)) != 0),
                "ZC_STD": np.std(np.diff(np.sign(window)) != 0),
                "VAR": np.var(window),
                "VAR_STD": np.std(window),
                "TD": np.sum(diff_abs_signal),
                "TD_STD": np.std(diff_abs_signal),
                "MAVS": np.mean(diff_abs_signal),
                "MAVS_STD": np.std(diff_abs_signal),
                "MNP": np.mean(window**2),
                "MNP_STD": np.std(window**2),
            }
            
            # Calculate slope sign changes
            ssc_values = (diff_signal[1:] * diff_signal[:-1]) < 0
            features["SSC"] = np.sum(ssc_values)
            features["SSC_STD"] = np.std(ssc_values)
            
            # Calculate coefficient of variation
            mean_signal = np.mean(window)
            features["CoV"] = (np.std(window) / mean_signal) if mean_signal != 0 else 0
            

        # Spectral metrics (still using CPU for now - can be moved to GPU if needed)
        freqs = np.fft.rfftfreq(len(window), d=1/fs)
        fft_magnitude = np.abs(np.fft.rfft(window))
        
        if np.sum(fft_magnitude) != 0:
            features["MNF"] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
            features["MNF_STD"] = np.std(freqs * fft_magnitude) / np.sum(fft_magnitude)
        else:
            features["MNF"] = 0
            features["MNF_STD"] = 0
            
        try:
            import pywt
            
            # Calculate appropriate wavelet decomposition level
            # The rule of thumb is level <= log2(signal_length)
            max_level = int(np.log2(len(window)))
            # Use a safe level (1-2 less than max to avoid boundary effects)
            safe_level = max(1, max_level - 2)
            
            # Use the determined safe level instead of hardcoded 4
            coeffs = pywt.wavedec(window, 'db4', level=safe_level)
            mdwt_values = np.array([np.sum(np.abs(c)) for c in coeffs])
            features["mDWT"] = np.sum(mdwt_values)
            features["mDWT_STD"] = np.std(mdwt_values)

        except (ImportError, ValueError) as e:
            # If wavelet calculation fails, log it and set values to 0
            # print(f"Warning: Wavelet calculation failed: {e}")
            features["mDWT"] = 0
            features["mDWT_STD"] = 0
        
                # Add frequency domain features
        freq_features = frequency_domain_features(window, fs)
        features.update(freq_features)
        
        # Add complexity features
        complex_features = complexity_features(window)
        features.update(complex_features)
        
        # Add advanced time domain features
        advanced_features = time_domain_advanced(window)
        features.update(advanced_features)
        
        return features
        
    except Exception as e:
        print(f"Error calculating features: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        # Return empty features including the new ones
        return {key: 0 for key in [
            # Original features
            "MAV", "MAV_STD", "IAV", "IAV_STD", "RMS", "RMS_STD", 
            "WL", "WL_STD", "ZC", "ZC_STD", "SSC", "SSC_STD", 
            "VAR", "VAR_STD", "CoV", "TD", "TD_STD", 
            "MAVS", "MAVS_STD", "MNF", "MNF_STD",
            "mDWT", "mDWT_STD", "MNP", "MNP_STD", "Kurt", "Kurt_STD",
            "MDF", "PKF", "MNP", "TTP", "SM1", "SM2", "SM3", "MNPF", # New features
            "SampEn", "CC", "LE", "HFD",
            "DASDV", "MYOP", "WAMP", "CARD", "LOG", "SKEW"
        ]}


def process_windows_in_batches(windows, channel, window_size, batch_size=64):
    """
    Process multiple windows in batches for GPU efficiency.
    
    Args:
        windows: List of DataFrame windows
        channel: Channel name to extract
        window_size: Expected window size
        batch_size: Number of windows to process at once
        
    Returns:
        List of dictionaries containing features for each window
    """
    all_features = []
    complete_windows = [w for w in windows if len(w) == window_size]
    
    if not complete_windows:
        return all_features
        
    if USE_GPU and batch_size > 1:
        # Process in batches for GPU efficiency
        for i in range(0, len(complete_windows), batch_size):
            batch = complete_windows[i:i+batch_size]
            
            # Extract signals from batch
            signals = [window[channel].values for window in batch]
            
            # Convert batch to tensors
            batch_tensors = torch.tensor(np.array(signals), dtype=torch.float32, device=DEVICE)
            
            # Process batch tensors
            batch_features = calculate_features_batch(batch_tensors)
            
            # Add window metadata and append to results
            for j, features in enumerate(batch_features):
                window_meta = {
                    "window_id": i + j,
                }
                window_data = {**window_meta, **features}
                all_features.append(window_data)
    else:
        # Process one by one (CPU mode or small batches)
        for i, window in enumerate(complete_windows):
            signal = window[channel].values
            features = calculate_features(signal)
            
            window_meta = {
                "window_id": i,
            }
            window_data = {**window_meta, **features}
            all_features.append(window_data)
            
    return all_features


def calculate_features_batch(batch_tensor, fs=2000):
    """
    Calculate features for a batch of windows using GPU acceleration.
    
    Args:
        batch_tensor: PyTorch tensor of shape [batch_size, window_length]
        fs: Sampling frequency in Hz
        
    Returns:
        List of feature dictionaries for each window
    """
    try:
        batch_size = batch_tensor.shape[0]
        features_list = []
        
        # Create absolute values and differences for the entire batch at once
        abs_signal = torch.abs(batch_tensor)
        diff_signal = batch_tensor[:, 1:] - batch_tensor[:, :-1]
        diff_abs_signal = torch.abs(diff_signal)
        
        # Calculate basic metrics efficiently across the batch
        mav = torch.mean(abs_signal, dim=1)
        mav_std = torch.std(abs_signal, dim=1)
        iav = torch.sum(abs_signal, dim=1)
        iav_std = torch.std(abs_signal, dim=1)
        rms = torch.sqrt(torch.mean(batch_tensor**2, dim=1))
        rms_std = torch.std(batch_tensor, dim=1)
        wl = torch.sum(diff_abs_signal, dim=1)
        wl_std = torch.std(diff_abs_signal, dim=1)
        var = torch.var(batch_tensor, dim=1)
        var_std = torch.std(batch_tensor, dim=1)
        td = torch.sum(diff_abs_signal, dim=1)
        td_std = torch.std(diff_abs_signal, dim=1)
        mavs = torch.mean(diff_abs_signal, dim=1)
        mavs_std = torch.std(diff_abs_signal, dim=1)
        mnp = torch.mean(batch_tensor**2, dim=1)
        mnp_std = torch.std(batch_tensor**2, dim=1)
        
        # Zero crossings (calculate for entire batch)
        zero_crossings = (batch_tensor[:, :-1] * batch_tensor[:, 1:]) < 0
        zc = torch.sum(zero_crossings, dim=1)
        zc_std = torch.std(zero_crossings.float(), dim=1)
        
        # Slope sign changes
        ssc_values = (diff_signal[:, :-1] * diff_signal[:, 1:]) < 0
        ssc = torch.sum(ssc_values, dim=1)
        ssc_std = torch.std(ssc_values.float(), dim=1)
        
        # Calculate for each window in the batch
        for i in range(batch_size):
            # Get all the pre-calculated metrics for this window
            features = {
                "MAV": mav[i].item(),
                "MAV_STD": mav_std[i].item(),
                "IAV": iav[i].item(),
                "IAV_STD": iav_std[i].item(),
                "RMS": rms[i].item(),
                "RMS_STD": rms_std[i].item(),
                "WL": wl[i].item(),
                "WL_STD": wl_std[i].item(),
                "ZC": zc[i].item(),
                "ZC_STD": zc_std[i].item(),
                "VAR": var[i].item(), 
                "VAR_STD": var_std[i].item(),
                "TD": td[i].item(),
                "TD_STD": td_std[i].item(),
                "MAVS": mavs[i].item(),
                "MAVS_STD": mavs_std[i].item(),
                "MNP": mnp[i].item(),
                "MNP_STD": mnp_std[i].item(),
                "SSC": ssc[i].item(),
                "SSC_STD": ssc_std[i].item(),
            }
            
            # Coefficient of variation
            mean_signal = torch.mean(batch_tensor[i]).item()
            features["CoV"] = (torch.std(batch_tensor[i]).item() / mean_signal) if mean_signal != 0 else 0
            
            
            # Move to CPU for remaining calculations
            window = batch_tensor[i].cpu().numpy()
            
            # Spectral metrics (on CPU)
            freqs = np.fft.rfftfreq(len(window), d=1/fs)
            fft_magnitude = np.abs(np.fft.rfft(window))
            
            if np.sum(fft_magnitude) != 0:
                features["MNF"] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
                features["MNF_STD"] = np.std(freqs * fft_magnitude) / np.sum(fft_magnitude)
            else:
                features["MNF"] = 0
                features["MNF_STD"] = 0
                
            # Fix for batch processing
            try:
                import pywt
                
                # Calculate appropriate wavelet decomposition level
                max_level = int(np.log2(len(window)))
                # Use a safe level (1-2 less than max to avoid boundary effects)
                safe_level = max(1, max_level - 2)
                
                # Use dynamic safe level instead of hardcoded 4
                coeffs = pywt.wavedec(window, 'db4', level=safe_level)
                mdwt_values = np.array([np.sum(np.abs(c)) for c in coeffs])
                features["mDWT"] = np.sum(mdwt_values)
                features["mDWT_STD"] = np.std(mdwt_values)
            except (ImportError, ValueError):
                features["mDWT"] = 0
                features["mDWT_STD"] = 0
                
            # Add frequency domain features
            freq_features = frequency_domain_features(window, fs)
            features.update(freq_features)
            
            # Add complexity features
            complex_features = complexity_features(window)
            features.update(complex_features)
            
            # Add advanced time domain features
            advanced_features = time_domain_advanced(window)
            features.update(advanced_features)
            
            features_list.append(features)
            
        return features_list
        
    except Exception as e:
        print(f"Error calculating batch features: {e}")
        traceback.print_exc()
        # Return empty features for each window in batch
        return [{key: 0 for key in ["MAV", "MAV_STD", "IAV", "IAV_STD", "RMS", "RMS_STD", 
                                  "WL", "WL_STD", "ZC", "ZC_STD", "SSC", "SSC_STD", 
                                  "VAR", "VAR_STD", "CoV", "TD", "TD_STD", 
                                  "MAVS", "MAVS_STD", "MNF", "MNF_STD",
                                  "mDWT", "mDWT_STD", "MNP", "MNP_STD", 
                                  "MDF", "PKF", "MNP", "TTP", "SM1", "SM2", "SM3", "MNPF",
                                  "SampEn", "CC", "LE", "HFD",
                                  "DASDV", "MYOP", "WAMP", "CARD", "LOG", "SKEW", "KURT"
                              ]}] * batch_tensor.shape[0]


def save_window_plots(windows, channel, subject, database, window_size, envelope_type, cutoff_freq, output_dir):
    """
    Save plots of the first 3 windows as images.
    
    Args:
        windows: List of windows to plot
        channel: EMG channel name
        subject: Subject identifier
        database: Database name
        window_size: Window size in samples
        envelope_type: Type of envelope used
        cutoff_freq: Filter cutoff frequency
        output_dir: Directory to save the plots
    """
    import matplotlib.pyplot as plt
    
    # Create subdirectory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot up to 3 windows
    for i, window in enumerate(windows[:3]):
        if len(window) != window_size:
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot the signal
        ax.plot(window.index, window[channel], 'b-')
        
        # Set labels and title
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        title_parts = [
            f"Window {i+1}",
            f"Subject {subject}",
            f"Channel {channel}",
            f"Win={window_size}",
            f"Env={envelope_type}",
            f"f={cutoff_freq}Hz"
        ]
        ax.set_title(' | '.join(title_parts))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate some stats to display in the figure
        stats = {
            'MAV': np.mean(np.abs(window[channel])),
            'Std': np.std(np.abs(window[channel])),
            'Max': np.max(np.abs(window[channel])),
            'RMS': np.sqrt(np.mean(window[channel]**2))
        }
        
        # Add stats text box
        stats_text = '\n'.join([f"{k}: {v:.4f}" for k, v in stats.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save figure
        filename = f"{database}_win{window_size}_env{envelope_type}_f{cutoff_freq}_subj{subject}_w{i}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close figure to avoid displaying
        
        #print(f"Saved window plot to {filepath}")


def process_file(file_path, subject, database, window_size, overlap, 
                 envelope_type, cutoff_freq, filtered_labels, channels=None):
    """
    Process a single .mat file and extract windowed features.
    
    Args:
        file_path: Path to the .mat file
        subject: Subject identifier
        database: Database name
        window_size: Window size in samples
        overlap: Overlap between windows in samples
        envelope_type: Type of envelope to apply (0=None, 1=Hilbert, 2=RMS, 3=MAV)
        cutoff_freq: Low-pass filter cutoff frequency in Hz
        filtered_labels: List of labels to include
        channels: List of channels to process (if None, only Channel 10 is processed)
        
    Returns:
        DataFrame containing features for each window
    """
    try:
        # Extract filename from path
        # Start timing
        start_time = time_module.time()
        timing = {}

        filename = os.path.basename(file_path)

        # Load file and build dataframe
        load_start = time_module.time()
        
        with suppress_output():
            # Load the .mat file
            mat_data = src.loadmatNina(database, filename, subject=subject)
            
            # Build dataframe
            test_df, grasps = src.build_dataframe(
                mat_file=mat_data,
                database=database,
                filename=filename,
                rectify=False,
                normalize=True
            )
        
        timing['load_data'] = time_module.time() - load_start
        #print(f"  - Data loaded in {timing['load_data']:.2f}s")

        # Filter and prepare channels
        prep_start = time_module.time()

        # Filter by labels if specified
        if filtered_labels:
            test_df = test_df[test_df['relabeled'].isin(filtered_labels)]
        # If no filtering is specified, process all labels
        else:
            test_df = test_df[test_df['relabeled'].notnull()]
            
        # Select channels to process
        if channels is None:
            # Process all channels
            emg_channels = [col for col in test_df.columns if 'Channel' in col]
            if emg_channels:
                channel_list = emg_channels
            else:
                print(f"Warning: No channels found in {filename}")
                return pd.DataFrame()
        elif channels == ['Channel 10'] and 'Channel 10' not in test_df.columns:
            # Specified Channel 10 but it's not in the data
            # Try to find any channel ending with 10
            channel_10_candidates = [col for col in test_df.columns if col.endswith('10')]
            if channel_10_candidates:
                channel_list = [channel_10_candidates[0]]
            else:
                print(f"Warning: Channel 10 not found in {filename}, using first available channel")
                emg_channels = [col for col in test_df.columns if 'Channel' in col]
                channel_list = [emg_channels[0]] if emg_channels else None
        else:
            channel_list = channels

        if not channel_list:
            print(f"Error: No suitable channels found in {filename}")
            return pd.DataFrame()
            
        all_features = []
        plots_saved = False  # Flag to track if we've already saved plots
        
        timing['preparation'] = time_module.time() - prep_start
        #print(f"  - Prep ready in {timing['preparation']:.2f}s")

        grasp_process_start = time_module.time()
        # Process each grasp/movement
        for grasp, grasp_df in test_df.groupby('stimulus'):
            if grasp_df.empty:
                continue
                
            # Get relabeled value (first occurrence)
            relabeled_value = grasp_df['relabeled'].iloc[0]
            
            # Extract metadata columns
            meta_columns = ["Time (s)", "subject", "re_repetition", "stimulus", "relabeled"]
            available_meta = [col for col in meta_columns if col in grasp_df.columns]
            
            # Process each channel
            for channel in channel_list:
                # Skip if channel not in dataframe
                if channel not in grasp_df.columns:
                    continue
                    
                # Apply envelope if specified
                if envelope_type > 0:
                    signal_df = src.get_envelope_lowpass(
                        grasp_df[[channel]], 
                        fm=2000,
                        cutoff_freq=cutoff_freq,
                        envelope_type=envelope_type
                    )
                else:
                    signal_df = grasp_df[[channel]]
                    
                # Create windows
                windows = src.create_windows_with_overlap(
                    signal_df,
                    window_size,
                    overlap
                )

                if not plots_saved and len(windows) > 0 and relabeled_value != 0:
                    # Create output directory for plots
                    output_dir = os.path.join('preprocessed_data', database)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save plots
                    save_window_plots(
                        windows, 
                        channel, 
                        subject, 
                        database, 
                        window_size, 
                        envelope_type, 
                        cutoff_freq, 
                        output_dir
                    )
                    plots_saved = True  # Set flag to avoid saving more plots
                
                # Get batch size from command line or use default
                batch_size = 64  # Default batch size
                
                # Process windows - either in batches (GPU) or individually (CPU)
                if USE_GPU and len(windows) > 1:
                    window_features = process_windows_in_batches(
                        windows, 
                        channel, 
                        window_size,
                        batch_size
                    )
                    
                    # Add metadata to each window
                    for i, features in enumerate(window_features):
                        window_meta = {
                            "subject": subject,
                            "filename": filename,
                            "grasp": grasp,
                            "relabeled": relabeled_value,
                            "channel": channel,
                            "window_id": i,
                        }
                        
                        # Combine metadata and features
                        window_data = {**window_meta, **features}
                        all_features.append(window_data)
                else:
                    # Original method - process each window individually
                    for i, window in enumerate(windows):
                        if len(window) == window_size:  # Only process complete windows
                            # Calculate features for this window
                            signal = window[channel].values
                            features = calculate_features(signal)
                            
                            # Add metadata
                            window_meta = {
                                "subject": subject,
                                "filename": filename,
                                "grasp": grasp,
                                "relabeled": relabeled_value,
                                "channel": channel,
                                "window_id": i,
                            }
                            
                            # Combine metadata and features
                            window_data = {**window_meta, **features}
                            all_features.append(window_data)
                
        
        timing["feature_extraction"] = time_module.time() - grasp_process_start
        #print(f"  - Features ready in {timing['feature_extraction']:.2f}s")
          
        # Create DataFrame from all features
        if all_features:
            #print(timing)
            return pd.DataFrame(all_features)
        else:
            print(f"No features extracted from {filename}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

        
def process_file_wrapper(args):
    """
    Wrapper function for parallel processing.
    Takes a tuple of arguments and passes them to process_file.
    
    Initializes CUDA within the worker process if needed.
    """
    # Unpack the GPU argument
    file_path, subject, database, window_size, overlap, envelope_type, cutoff_freq, filtered_labels, channels, use_gpu = args
    
    # Initialize CUDA in the worker process if needed
    global USE_GPU, DEVICE
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                USE_GPU = True
                DEVICE = torch.device("cuda")
            else:
                USE_GPU = False
        except:
            USE_GPU = False
    else:
        USE_GPU = False
    
    # Call the main processing function
    return process_file(file_path, subject, database, window_size, overlap, envelope_type, cutoff_freq, filtered_labels, channels)


def main():
    """Main function to process all files in a database."""
    # Parse command-line arguments
    args = parse_arguments()
    
     # Extract parameters from arguments
    database = args.database
    window_size = args.window_size
    overlap = args.overlap
    envelope_type = args.envelope_type
    cutoff_freq = args.filter_cutoff
    use_gpu = args.use_gpu
    
    # Check if GPU is available in main process (for information only)
    if use_gpu:
        has_gpu = setup_cuda(False)
        if not has_gpu:
            print("Warning: GPU acceleration requested but not available")
            print("Will try to initialize CUDA in worker processes")
    else:
        print("GPU acceleration not requested, using CPU")

    # Convert filtered labels if specified, otherwise use empty list to process all labels
    filtered_labels = []
    if args.filtered_labels.strip():
        filtered_labels = [int(label.strip()) for label in args.filtered_labels.split(',')]
    
    # Determine channels to process
    if args.only_channel_10:
        channels = ['Channel 10']  # Only process Channel 10
    else:
        channels = None  # Process all available channels
        
    # Create output filename based on parameters
    output_file = f"{database}_w{window_size}_env{envelope_type}_f{cutoff_freq}"
    
    print(f"Processing database: {database}")
    print(f"Window size: {window_size} samples ({window_size/2000:.3f} seconds at 2kHz)")
    print(f"Overlap: {overlap} samples")
    print(f"Envelope type: {envelope_type}")
    print(f"Filter cutoff: {cutoff_freq} Hz")
    print(f"Labels: {'All' if not filtered_labels else filtered_labels}")
    print(f"Channels: {'Only Channel 10' if args.only_channel_10 else 'All available channels'}")
    print(f"Output file will be: {output_file}")
        
    # Path to the database folder
    data_path = os.path.abspath(os.path.join('data', database))
    
    # Find all subject folders
    subjects = []
    for item in os.listdir(data_path):
        if re.match(r'[sS]\d+', item) or re.match(r'Subject\d+', item):
            subjects.append(item)
    
    if not subjects:
        print(f"No subject folders found in {data_path}")
        return
        
    print(f"Found {len(subjects)} subject folders.")
    
    # Prepare arguments for parallel processing
    all_args = []
    files_by_subject = {}
    
    for subject in subjects:
        subject_path = os.path.join(data_path, subject)
        mat_files = [f for f in os.listdir(subject_path) if f.endswith('.mat')]
        files_by_subject[subject] = mat_files
        
        for filename in mat_files:
            file_path = os.path.join(subject_path, filename)
            all_args.append((
                file_path, subject, database, window_size, overlap, 
                envelope_type, cutoff_freq, filtered_labels, channels, use_gpu
            ))
    
    total_files = len(all_args)
    print(f"Total .mat files to process: {total_files}")

    # Determine number of processes to use (up to 80% of available cores)
    num_processes = max(1, min(cpu_count() - 1, int(cpu_count() * 0.8)))
    print(f"Using {num_processes} parallel processes")
    
    # Initialize result dataframe and tracking variables
    all_results = pd.DataFrame()
    start_time = time.time()
    processed_windows = 0

    # Error tolerance settings
    max_errors = int(total_files * 0.1)  # Allow up to 10% of files to fail
    error_count = 0
    failed_files = []

    # Memory efficiency settings
    max_rows_before_save = 500000  # Save every 500k windows to manage memory

    # Create output directory in preprocessed_data folder
    output_dir = os.path.join('preprocessed_data', database)
    os.makedirs(output_dir, exist_ok=True)

    # Process files in parallel with error tolerance and memory management
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress tracking
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for i, (args_tuple, result_df) in enumerate(zip(all_args, pool.imap(process_file_wrapper, all_args))):
                file_path = args_tuple[0]
                file_name = os.path.basename(file_path)
                
                # Check if processing succeeded
                if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                    # Success - add to results
                    all_results = pd.concat([all_results, result_df], ignore_index=True)
                    processed_windows += len(result_df)
                    
                    # Check if we should save intermediate results to manage memory
                    if len(all_results) >= max_rows_before_save:
                        # Save intermediate chunk
                        chunk_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        chunk_filename = os.path.join(output_dir, f"chunk_{output_file}_{chunk_timestamp}.parquet")
                        
                        try:
                            all_results.to_parquet(chunk_filename, index=False, engine='pyarrow')
                            #print(f"\nIntermediate chunk saved: {len(all_results)} windows to {chunk_filename}")
                            
                            # Clear dataframe to free memory
                            all_results = pd.DataFrame()
                        except Exception as e:
                            print(f"\nWarning: Could not save intermediate chunk: {e}")
                else:
                    # Processing failed
                    error_count += 1
                    failed_files.append(file_name)
                    
                    # Check if we've exceeded error threshold
                    if error_count > max_errors:
                        print(f"\nToo many errors ({error_count}/{i+1}). Stopping processing.")
                        break
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix(windows=processed_windows, errors=error_count)
                    
    # Save results
    if not all_results.empty or os.path.exists(output_dir):
        # Find any previously saved chunks
        chunks = [f for f in os.listdir(output_dir) if f.startswith(f"chunk_{output_file}_")]
        
        # Create a list to collect all dataframes
        all_dfs = []
        total_windows = processed_windows
        
        # If we have chunks, load and combine them
        if chunks:
            print(f"\nFound {len(chunks)} previously saved chunks. Consolidating...")
            
            # Load all chunks
            for i, chunk_file in enumerate(chunks):
                chunk_path = os.path.join(output_dir, chunk_file)
                try:
                    print(f"Loading chunk {i+1}/{len(chunks)}: {chunk_file}")
                    chunk_df = pd.read_parquet(chunk_path)
                    all_dfs.append(chunk_df)
                    total_windows += len(chunk_df)
                    print(f"  - {len(chunk_df)} windows loaded")
                except Exception as e:
                    print(f"Warning: Could not read chunk {chunk_file}: {e}")
        
        # Add current results if not empty
        if not all_results.empty:
            all_dfs.append(all_results)
        
        # Combine all dataframes if we have any
        if all_dfs:
            print(f"\nCombining {len(all_dfs)} dataframes with a total of {total_windows} windows...")
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            final_filename = os.path.join(output_dir, output_file + f"[{timestamp}].parquet")
            
            print(f"Saving consolidated file with {len(combined_df)} windows to {final_filename}")
            
            try:
                # Save to parquet format
                combined_df.to_parquet(final_filename, index=False, engine='pyarrow')
                print(f"Consolidated file successfully saved to {final_filename}")
                
                # Delete intermediate chunks after successful consolidation
                if chunks:
                    print("Cleaning up intermediate chunks...")
                    for chunk_file in chunks:
                        try:
                            os.remove(os.path.join(output_dir, chunk_file))
                            print(f"  - Deleted: {chunk_file}")
                        except Exception as e:
                            print(f"  - Could not remove chunk {chunk_file}: {e}")
            except Exception as e:
                print(f"Error saving consolidated parquet: {e}")
                # Fallback to CSV if parquet fails
                csv_filename = final_filename.replace('.parquet', '.csv')
                print(f"Attempting to save as CSV instead: {csv_filename}")
                combined_df.to_csv(csv_filename, index=False)
                print("CSV file saved successfully.")
            
            # Print summary stats
            total_time = time.time() - start_time
            print(f"\nProcessing complete in {total_time/60:.2f} minutes!")
            print(f"Average speed: {total_files / max(1, total_time):.2f} files/second")
            print(f"Total windows extracted: {len(combined_df)}")
        else:
            print("No data was processed. Check your input parameters.")
    else:
        print("No data was processed. Check your input parameters.")
        

if __name__ == "__main__":
    main()