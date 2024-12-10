import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import src.db_utils as db_utils
import src.preprocessing_utils as prep_utils
from src.config import DATABASE_INFO

def plot_data(filtered_emg_data, restimulus_data, grasp_number=None, interactive=False, frequency=None):
    emg_df = pd.DataFrame(filtered_emg_data, columns=[f'Channel {i+1}' for i in range(filtered_emg_data.shape[1])])

    if frequency is not None:
        time = [i / frequency for i in range(len(emg_df))]
        emg_df['Time (s)'] = time
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

def plot_emg_data(database, mat_file, grasp_number, interactive=False, time=True, include_rest=False, padding = 10, use_stimulus = False):
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