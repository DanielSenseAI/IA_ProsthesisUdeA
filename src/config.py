from src import process_data as data_process

URLS = {
    1:"https://ninapro.hevs.ch/files/DB1/Preprocessed/s", #1.zip
    2:"https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s", #1.zip
    3:"https://ninapro.hevs.ch/files/db3_Preproc/s" #1_0.zip
}

OUTPUT_DIRS = {
    1:"signals/DB1",
    2:"signals/DB2",
    3:"signals/DB3"
}

SUBJECTS = {
    1:27,
    2:40,
    3:11
}

DATABASES = {
    'DB1': 2000,
    'DB2': 2000,
    'DB3': 2000
}#Databases with sampling frequencies in Hz

FEATURES = {
    'RMS_E': data_process.calculate_rms,
    'MAV_E': data_process.calculate_mav, 
    # 'MAVS_E': data_process.calculate_mavs,
    'VARIANCE_E': data_process.calculate_variance,
    'SAMPLE_VARIANCE_E': data_process.calculate_sample_variance,
    # 'KURTOSIS_E': data_process.calculate_kurtosis,
    # 'SKEWNESS_E': data_process.calculate_skewness
}

MOVEMENTS_LABEL = {
    '0' : 'Base',
    '17': 'Lateral',
    '9' : 'Writing Tripod',
    '5' : 'Medium Wrap',
    '13': 'Tripod',
    '10': 'Power Sphere',
    '6' : 'Ring', 
}

FC= 10 #Cutoff frequency in Hz
WINDOWING = 0.03 #s time of the window // Ej: 0.2s = 200 ms
OVERLAPPING = 50 #% overlapping percentage
REPETITIONS = 10 #Number of times a movement is repeated
ELECTRODES = 8 #Number of electrodes used in the experiment
THRESHOLD = 60 # Characterization percentage