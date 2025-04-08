# IA_ProsthesisUdeA

# AI Prosthesis Models Development

This repository focuses on developing AI models for prosthetic control through EMG signal analysis. It is part of a collaborative investigation between the University of Antioquia and various partners, with special thanks to Protesis Avanzadas SAS, a Colombian company ([www.protesisavanzadas.com](http://www.protesisavanzadas.com)).

---

## Project Overview

This project implements a comprehensive EMG signal processing and feature extraction pipeline for use in prosthetic device control. The system processes raw EMG signals from datasets like Ninapro, extracts meaningful features, and prepares data for machine learning models that can classify gestures and movements.

Key capabilities:
- Multi-threaded processing of large EMG datasets
- Comprehensive feature extraction (time domain, frequency domain, and complexity metrics)
- GPU acceleration for faster processing
- Configurable preprocessing parameters (window sizes, overlap, envelopes, filters)
- Grid search for optimal parameter selection
- Memory-efficient handling of large datasets

---

## Project Structure

Notebooks in this project are to be run from the root folder. All routes are relative and consider that there is a "data" folder in which the EMG database files are organized.

- **`model/`**: Contains AI or machine learning models under development.
- **`preprocessed_data/`**: Datasets prepared for training and evaluation.
- **`src/`**: Core source code, including:
  - `config.py`: Select databases, subjects, features and others.
  - `download_data_utils.py`: Download Ninapro DBs on demand.
  - `model_utils.py`: Useful model callbacks and other utilities.
  - `preprocessing_utils.py`: Signal segmentation, filters, and labeling.
  - `process_data.py`: Features to be extracted.

### Main Scripts

- **`wf_extractor.py`**: Main feature extraction script that processes EMG signals and extracts features.
  - Processes EMG data from .mat files in a database
  - Applies various preprocessing techniques
  - Extracts comprehensive feature sets
  - Supports GPU acceleration with PyTorch
  - Memory-efficient processing for large datasets

- **`mass_wf_extractor.py`**: Grid search utility to find optimal preprocessing parameters.
  - Systematically tests combinations of window sizes, envelope types, and filter cutoffs
  - Provides visualization and logging of results

- **`prot_avanzadas.ipynb`**: A Jupyter notebook documenting initial analyses and experiments.
- **`requirements.txt`**: Specifies libraries to be used for the project.

---

## EMG Feature Extraction

The system extracts a comprehensive set of features from EMG signals, including:

### Time-Domain Features
- **MAV**: Mean Absolute Value - average of absolute signal values
- **RMS**: Root Mean Square - square root of average power
- **WL**: Waveform Length - cumulative length of the waveform
- **ZC**: Zero Crossings - frequency of signal crossing zero
- **SSC**: Slope Sign Changes - frequency of direction changes
- **VAR**: Variance - signal power variation
- **MAVS**: Mean Absolute Value Slope - rate of change in MAV
- **DASDV**: Difference Absolute Standard Deviation Value
- **MYOP**: Myopulse Percentage Rate - muscle activation percentage
- **WAMP**: Willison Amplitude - threshold crossings in signal differences
- **LOG**: Log Detector - exponential of mean log values
- **SKEW**: Skewness - asymmetry of signal distribution
- **KURT**: Kurtosis - peakedness of signal distribution

### Frequency-Domain Features
- **MDF**: Median Frequency - frequency dividing power spectrum in half
- **PKF**: Peak Frequency - frequency with maximum power
- **MNF**: Mean Frequency - weighted average frequency
- **TTP**: Total Power - sum of power spectrum values
- **SM1/SM2/SM3**: Spectral Moments - distribution of frequency content
- **MNPF**: Mean Power Frequency - weighted frequency average

### Wavelet Features
- **mDWT**: Modified Discrete Wavelet Transform features
- **mDWT_STD**: Standard deviation of wavelet coefficients

### Complexity Features
- **SampEn**: Sample Entropy - signal predictability/complexity
- **CC**: Correlation Dimension - dimensionality of the system
- **LE**: Lyapunov Exponent estimate - chaos measurement
- **HFD**: Higuchi Fractal Dimension - signal complexity

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/DanielSenseAI/IA_ProsthesisUdeA.git
   cd IA_ProsthesisUdeA
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Organize databases for local use:
   - Create a folder called "data" on the root folder (where this README is located)
   - Inside this folder DB1 to DB10 correspond to the Ninapro Databases
      - Create a folder called "DB{number}" for the corresponding database
      - Subject data can be separated in different folders for each user named s1 to s{number}
   - Further DBs will be added from different sources

5. The 'config' file may be modified to include or exclude processes and modify general settings for each process

---

## Using Feature Extraction Scripts

### Basic Feature Extraction

To extract features from an EMG database:

```bash
python wf_extractor.py --database DB4 --window-size 200 --envelope-type 1 --filter-cutoff 1.0
```

#### Parameters:
- `--database`: Database name (e.g., DB4)
- `--window-size`: Window size in samples (200 = 100ms at 2kHz)
- `--overlap`: Overlap between windows in samples
- `--envelope-type`: Type of envelope (0=None, 1=Hilbert, 2=RMS, 3=MAV)
- `--filter-cutoff`: Low-pass filter cutoff frequency in Hz
- `--use-gpu`: Enable GPU acceleration (requires PyTorch)
- `--batch-size`: Batch size for GPU processing
- `--only-channel-10`: Process only Channel 10 instead of all channels
- `--filtered-labels`: Comma-separated list of labels to include

### Grid Search for Optimal Parameters

To find optimal preprocessing parameters:

```bash
python mass_wf_extractor.py --database DB4 --window-sizes 100,200,400 --envelope-types 0,1,2 --filter-cutoffs 0.6,1,5 --use-gpu
```

#### Parameters:
- `--database`: Database name
- `--window-sizes`: Comma-separated list of window sizes to try
- `--envelope-types`: Comma-separated envelope types (0=None, 1=Hilbert, 2=RMS, 3=MAV)
- `--filter-cutoffs`: Comma-separated cutoff frequencies
- `--overlap-pct`: Percentage of overlap between windows
- `--use-gpu`: Enable GPU acceleration
- `--batch-size`: Batch size for GPU processing
- `--python-executable`: Path to specific Python executable
- `--resume`: Resume from last completed combination

---

## Output Data Format

The extracted features are saved in parquet files with the following structure:

- Each row represents one window of EMG data
- Columns include metadata (subject, grasp type, channel) and all extracted features
- Files are named according to the parameters used for extraction

---

## Development and Contribution

This project is under active development as part of a PhD research program. Contributions are welcome through pull requests or issues. Please contact the repository maintainer for major changes.

---

## License

This project is provided for research purposes. Please check with the repository maintainer for usage rights.

---

## Acknowledgments

Special thanks to Protesis Avanzadas SAS and the University of Antioquia for their support in this research.