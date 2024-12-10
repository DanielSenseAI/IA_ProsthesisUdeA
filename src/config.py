from src import process_data as data_process

FC= 10 #Cutoff frequency in Hz
ELECTRODES = 8 #Number of electrodes used in the experiment  // Incrementar a 10
THRESHOLD = 60 # Characterization percentage

WINDOWING = [0.01, 0.02, 0.3]  # s time of the window // Ej: 0.2s = 200 ms
OVERLAPPING = [0, 33, 50]  # % overlapping percentage

URLS = {
    1:"https://ninapro.hevs.ch/files/DB1/Preprocessed/s", #1.zip
    2:"https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s", #1.zip
    3:"https://ninapro.hevs.ch/files/db3_Preproc/s" #1_0.zip
}

OUTPUT_DIRS = {
    1:"Ninapro/DB1",
    2:"Ninapro/DB2",
    3:"Ninapro/DB3",
    4:"Ninapro/DB4",
    5:"Ninapro/DB5",
    6:"Ninapro/DB6",
    7:"Ninapro/DB7",
    8:"Ninapro/DB8",
    9:"Ninapro/DB9",
}

DATABASE_INFO = {
    'DB1': {'frequency': 100, 'subjects': 27, 'repetitions': 10, 'electrodes': 10},
    'DB2': {'frequency': 2000, 'subjects': 40, 'repetitions': 6, 'electrodes': 12},
    'DB3': {'frequency': 2000, 'subjects': 11, 'repetitions': 6, 'electrodes': 12},
    'DB4': {'frequency': 2000, 'subjects': 10, 'repetitions': 6, 'electrodes': 12},
    'DB5': {'frequency': 200,  'subjects': 10, 'repetitions': 6, 'electrodes': 16},
    'DB6': {'frequency': 2000, 'subjects': 10, 'repetitions': 6, 'electrodes': 16}, #two columns are empty
    'DB7': {'frequency': 2000, 'subjects': 22, 'repetitions': 6, 'electrodes': 12},
    'DB8': {'frequency': 2000, 'subjects': 12, 'repetitions': 6, 'electrodes': 16},
    'DB9': {'frequency': 2000, 'subjects': 77, 'repetitions': 6, 'electrodes': 0},
}

FEATURES = {
    'RMS_E': data_process.calculate_rms,
    'MAV_E': data_process.calculate_mav, 
    'MAVS_E': data_process.calculate_mavs,
    'VARIANCE_E': data_process.calculate_variance,
    'SAMPLE_VARIANCE_E': data_process.calculate_sample_variance,
    'KURTOSIS_E': data_process.calculate_kurtosis,
    'SKEWNESS_E': data_process.calculate_skewness
}

MOVEMENTS_LIBRARY = {
    '0' : 'Rest',
    '1' : 'Thumb up',
    '2' : 'Extension index and middle',
    '3' : 'Flexion ring and little',
    '4' : 'Thumb down',
    '5' : 'Abduction all fingers',
    '6' : 'Fist',
    '7' : 'Pointing Index',
    '8' : 'Adduction extended fingers',
    '9' : 'Wrist supination MF',
    '10': 'Wrist pronation MF',
    '11': 'Wrist supination LF',
    '12': 'Wrist pronation LF',
    '13': 'Wrist flexion',
    '14': 'Wrist extension',
    '15': 'Wrist radial deviation',
    '16': 'Wrist ulnar deviation',
    '17': 'Wrist extension hand closed',
    '18': 'Large diameter',
    '19': 'Small diameter',
    '20': 'Fixed hook',
    '21': 'Index finger extension',
    '22': 'Medium Wrap',
    '23': 'Ring', 
    '24': 'Prismastic four fingers',
    '25': 'Stick',
    '26': 'Writing Tripod',
    '27': 'Power Sphere',
    '28': 'Three finger sphere',
    '29': 'precision sphere',
    '30': 'Tripod',
    '31': 'Prismatic pinch',
    '32': 'Tip pinch',
    '33': 'Quadpod',
    '34': 'Lateral',
    '35': 'Parallel Extension',
    '36': 'Extension type',
    '37': 'Power Disk',
    '38': 'Bottle Tripod Grasp',
    '39': 'Turn a screwdriver',
    '40': 'Cut something-knife', 
    '41': 'StrFlexion little',
    '42': 'StrFlexion ring',
    '43': 'StrFlexion middle',
    '44': 'StrFlexion index',
    '45': 'StrAbduction thumb',
    '46': 'StrFlexion thumb',
    '47': 'StrFlexion Index and Little',
    '48': 'StrFlexion Ring and middle',
    '49': 'StrExtension Index and Thumb',
    '50': 'Index flexion',
    '51': 'Index Extension',
    '52': 'Middle flexion',
    '53': 'Middle Extension',
    '54': 'Ring flexion',
    '55': 'Ring Extension',
    '56': 'Little flexion',
    '57': 'Little Extension',
    '58': 'Thumb flexion',
    '59': 'Thumb Adduction',
    '60': 'Thumb Adbuction',
    '61': 'Thumb Extension',
}

MOVEMENTS_LABEL_DEPRECIATED = {
    '0' : 'Base',
    '17': 'Lateral',
    '9' : 'Writing Tripod',
    '5' : 'Medium Wrap',
    '13': 'Tripod',
    '10': 'Power Sphere',
    '6' : 'Ring', 
}
