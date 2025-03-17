import io
import os
import zipfile

import requests

from src import config

 
def download_data(url: str, database: int):
    """
    Función para descar la información de cada URL
    Args:
        - url (str): Url de la descarga
        - database (int): Base de datos a la que hace referencia
    Returns:
        - None
    """
    OUTPUT_DIR = config.OUTPUT_DIRS.get(database)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Comenzando descarga de {url}')
    response = requests.get(url)
    if response.status_code == 200:
        print('La información fue extraida correctamente: ')
        zip_file = zipfile.ZipFile(io.BytesIO(response.content)) 
        zip_file.extractall(OUTPUT_DIR)
        print(f"Archivos extraídos en: {OUTPUT_DIR}")
    else:
        print(f"Error al descargar el archivo {url}")


def prepare_download_data(database: int):
    """
    Función para descargar la data referente a Nina pro
    Args: 
        - database (int): Base de datos que se desea descargar
    Returns:
        - None 
    """
    subjects = config.SUBJECTS.get(database)
    urls = [
        f"{config.URLS.get(database)}{i}_0.zip" if database == 3 else f"{config.URLS.get(database)}{i}.zip"
        for i in range(1, subjects + 1)
    ]
    print(f'URLS: {urls}')
    list(map(lambda url: download_data(url, database), urls))


def get_file_path_database(database: str) -> list:
    """
        Busca y devuelve las rutas de archivos `.mat` dentro de una base de datos de señales, filtrando solo aquellos archivos que contienen 'E3' en su nombre.

        Parámetros:
        ----------
        database : str
            El nombre de la base de datos. Este valor se utiliza para construir la ruta principal donde se almacenan las señales.

        Retorna:
        -------
        list
            Una lista de cadenas que contienen las rutas completas de los archivos `.mat` que cumplen con los criterios de búsqueda (contienen 'E3' en el nombre del archivo).
    """
    main_folder_path = f'data/{database}/'
    signal_paths = []
    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)  
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and 'E3' in file and file.endswith('.mat'):
                    signal_paths.append(file_path)
    return signal_paths


