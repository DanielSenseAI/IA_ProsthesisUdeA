import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import src.preprocessing_utils as prep_utils
from src.config import DATABASE_INFO
from src import *

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
    total_test_time = (emg_count/database_info['electrodes']) / database_info['frequency']
    unique_restimulus = np.unique(mat_file['restimulus'])

    print(f"Loaded file: {filename}")
    print(f"Total test time: {total_test_time} seconds")
    print(f"Total test time: {total_test_time/60} minutes")
    print(f"Unique restimulus values: {unique_restimulus}")
    print(f"Total EMG samples: {emg_count}")
    print(f"Frequency: {database_info['frequency']} Hz")

    print("Summary for the whole file:")
    print(summary_df)
    return mat_file

def extract_data(mat_file, use_Stimulus=False):
    emg_data = mat_file.get('emg', None)
    if use_Stimulus:
        restimulus_data = mat_file.get('stimulus', None)
    else:
        restimulus_data = mat_file.get('restimulus', None)
    
    if emg_data is None:
        print("No 'emg' data found in the file.")
        return None, None
    
    if restimulus_data is None:
        print("No 'restimulus' data found in the file.")
        return None, None
    
    return emg_data, restimulus_data

def build_dataframe(mat_file, database, filename, use_Stimulus=False, rectify=False, normalize=False):

    database_info = DATABASE_INFO[database]

    emg_data, restimulus_data = extract_data(mat_file, use_Stimulus)
    subject = mat_file.get('subject') or mat_file.get('subj')
    re_repetition = mat_file.get('rerepetition', None)
    excercise = get_exercise_number(mat_file, filename)

    emg_df = pd.DataFrame(emg_data, columns=[f'Channel {i+1}' for i in range(emg_data.shape[1])])
    if normalize:
        emg_df = (emg_df - emg_df.mean()) / emg_df.std()
    if rectify:
        emg_df = emg_df.abs()

    emg_df = prep_utils.add_time(emg_df, database_info['frequency'])
    emg_df["subject"] = subject if isinstance(subject, list) else [subject] * len(emg_df)
    emg_df["re_repetition"] = re_repetition
    emg_df["stimulus"] = restimulus_data
    emg_df = prep_utils.relabel_database(database, emg_df, exercise = excercise)   
    unique_restimulus = np.unique(mat_file['restimulus'])
    print(f"Unique restimulus values: {unique_restimulus}")
    print(f"New restimulus values in Relabeled: {emg_df['relabeled'].unique()}")	 
    
    return emg_df, unique_restimulus

def get_exercise_number(mat_file, filename=None):
    # Get the exercise from the name of the mat_file by searching the convention "E" + number
    exercise = None
    
    # Split the name by "_"
    parts = filename.split("_")
    
    # Use a regular expression to find the part that matches "E" followed by a number
    for part in parts:
        match = re.match(r'E(\d+)', part)
        if match:
            exercise = int(match.group(1))
            break
    
    return exercise

def append_to_dataframe(mat_file, database, df, use_Stimulus=False):
    new_df = build_dataframe(mat_file, database, use_Stimulus)
    return pd.concat([df, new_df], ignore_index=True)

def filter_data(emg_data, restimulus_data, chosen_number, include_rest=False, padding=0):
    chosen_indices = np.where(restimulus_data == chosen_number)[0]
    
    if chosen_indices.size == 0:
        print(f"No data found for the chosen number: {chosen_number}")
        return None
    
    first_index = max(0, chosen_indices[0] - padding)
    last_index = min(len(restimulus_data) - 1, chosen_indices[-1] + padding)
    
    if include_rest:
        filtered_indices = np.where((restimulus_data[first_index:last_index + 1] == chosen_number) | 
                                    (restimulus_data[first_index:last_index + 1] == 0))[0] + first_index
        print("Rest included in the movement extraction!")
    else:
        filtered_indices = np.where(restimulus_data[first_index:last_index + 1] == chosen_number)[0] + first_index
        print("Extracting data without rest!")
    
    filtered_emg_data = emg_data[filtered_indices, :]
    filtered_restimulus_data = restimulus_data[filtered_indices]
    
    if filtered_emg_data.size == 0:
        unique_restimulus = np.unique(restimulus_data)
        print(f"No EMG data found for the chosen number: {chosen_number}")
        print(f"Unique items in restimulus: {unique_restimulus}")
        return None
    
    return filtered_emg_data, filtered_restimulus_data

def filter_data_pandas(emg_data_df, chosen_number, restimulus_column='relabeled', include_rest=False, padding=0):
    restimulus_data = emg_data_df[restimulus_column].values
    chosen_indices = np.where(restimulus_data == chosen_number)[0]
    
    if chosen_indices.size == 0:
        print(f"No data found for the chosen number: {chosen_number}")
        return None
    
    first_index = max(0, chosen_indices[0] - padding)
    last_index = min(len(restimulus_data) - 1, chosen_indices[-1] + padding)
    
    if include_rest:
        filtered_indices = np.where((restimulus_data[first_index:last_index + 1] == chosen_number) | 
                                    (restimulus_data[first_index:last_index + 1] == 0))[0] + first_index
        print("Rest included in the movement extraction!")
    else:
        filtered_indices = np.where(restimulus_data[first_index:last_index + 1] == chosen_number)[0] + first_index
        print("Extracting data without rest!")
    
    filtered_emg_data = emg_data_df.iloc[filtered_indices, :]
    
    if filtered_emg_data.size == 0:
        unique_restimulus = emg_data_df[restimulus_column].unique()
        print(f"No EMG data found for the chosen number: {chosen_number}")
        print(f"Unique items in restimulus: {unique_restimulus}")
        return None
    
    return filtered_emg_data
