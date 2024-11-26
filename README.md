# IA_ProsthesisUdeA
---

# AI Prosthesis Models Development  

This repository focuses on developing AI models for prosthetic control. It is part of a collaborative investigation between the University of Antioquia and various partners, with special thanks to Protesis Avanzadas SAS, a Colombian company ([www.protesisavanzadas.com](http://www.protesisavanzadas.com)).  

---

## Project Structure  

Notebooks in this project are to be run from the root folder. All routes are relative and consider that there is a "Ninapro" folder (not uploaded with the Repo) in which the Ninapro Files have been decompressed.

- **`model/`**: Contains AI or machine learning models under development.  
- **`preprocessed_data/`**: Datasets prepared for training and evaluation. Contains feature data ideally, to be used for model training and evaluation.  
- **`src/`**: Core source code, including:  
  - config.py: Select databases, subjects, features and others.  
  - download_data_utils.py: Download Ninapro DBs on demand. 
  - model_utils.py: useful model callbacks and others. 
  - preprocessing_utils.py: signal segmentation, filters, labeling.
  - process_data.py: features to be extracted. 

- **`prot_avanzadas.ipynb`**: A Jupyter notebook documenting initial analyses and experiments conducted with Carlos Muñoz, UNIR and Protesis Avanzadas.  
- **`requirements.txt`**: Specifies libraries to be used for the project.  

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

--- 
