import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from src import config
from src.config import DATABASE_INFO

import os
from scipy.io import loadmat
import pandas as pd
import numpy as np


def loadmatNina(database, filename=None, subject="s1"):
    """
    Load a .mat file and return its contents as a dictionary.
    """
    # Define the path to the subject's folder
    subject_folder = f'data/{database}/{subject}/'
    try:
        database_info = DATABASE_INFO[database]
    except KeyError:
        raise ValueError(f"Database '{database}' not found in DATABASE_INFO.")
    
    # If no filename is provided, find the first .mat file in the folder
    if filename is None:
        for file in os.listdir(subject_folder):
            if file.endswith('.mat'):
                filename = file
                break
        if filename is None:
            raise FileNotFoundError("No .mat file found in the specified directory.")
    
    # Open the .mat file and show a summary for the whole file
    sample_file = os.path.join(subject_folder, filename)
    mat_file = loadmat(sample_file)

    # Create a DataFrame for the summary 
    summary_data = {}
    for key, value in mat_file.items():
        if isinstance(value, np.ndarray):
            summary_data[key] = pd.Series(value.flatten()).describe()
        elif isinstance(value, (list, tuple)):
            summary_data[key] = pd.Series(np.array(value).flatten()).describe()

    summary_df = pd.DataFrame(summary_data)

    # Calculate the total test time using the count from summary data
    emg_count = summary_df.loc['count', 'emg']
    total_test_time = emg_count / database_info['frequency']

    print(f"Loaded file: {filename}")
    print(f"Total test time: {total_test_time} seconds")
    print(f"Total test time: {total_test_time/60} minutes")
    print(f"Total EMG samples: {emg_count}")
    print(f"Frequency: {database_info['frequency']} Hz")

    print("Summary for the whole file:")
    print(summary_df)
    return mat_file

def extract_data(mat_file):
    emg_data = mat_file.get('emg', None)
    restimulus_data = mat_file.get('restimulus', None)
    
    if emg_data is None:
        print("No 'emg' data found in the file.")
        return None, None
    
    if restimulus_data is None:
        print("No 'restimulus' data found in the file.")
        return None, None
    
    return emg_data, restimulus_data

def filter_data(emg_data, restimulus_data, chosen_number):
    filtered_indices = np.where(restimulus_data == chosen_number)[0]
    filtered_emg_data = emg_data[filtered_indices, :]
    filtered_restimulus_data = restimulus_data[filtered_indices]
    
    if filtered_emg_data.size == 0:
        unique_restimulus = np.unique(restimulus_data)
        print(f"No EMG data found for the chosen number: {chosen_number}")
        print(f"Unique items in restimulus: {unique_restimulus}")
        return None
    
    return filtered_emg_data, filtered_restimulus_data
