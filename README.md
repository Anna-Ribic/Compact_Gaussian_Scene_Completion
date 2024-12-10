
# General Scaffold-GS Reconstruction

This repository provides a training procedure for reconstruction multiple scenes with fixed attribute mlps using the Scaffold-GS approach or a single-Gaussian anchor based approach inspired by Scaffold-GS

## Data
See how to create a source data folder with scene image set and sparse reconstruction at https://github.com/Anna-Ribic/BlenderProc

## Reconstruction Pipeline

1. Optimize Single Gaussians Scene Representation
   - **Usage**:
     ```bash
     python opt_singleGaussians.py --config config_opt_singleGaussians.json
     ```
   - **Arguments**:
     - `--config`: Path to the JSON configuration file for single Gaussian optimization.
    
    - **`train_files`**: Specifies the file containing the scenes to be optimized
    - **`exp_name`**: Name of the experiment
    - **`writer_name`**: Identifier for the logging writer
    - **`save_path`**: Directory where results will be saved
    - **`dataset_config.data_path`**: Path to the dataset created in the 'Data' step
  

2. Train or Evaluate Completion Model
    - **Usage**:
     ```bash
     python completion_singleGaussian.py config_singleGaussian.json (--evaluate)
     ```
   - **Arguments**:
     - `--config`: Path to the JSON configuration file for scene completion.
   
    - **`file_names.train`**: Specifies the file containing training scenes names
    - **`file_names.val`**: Specifies the file containing test scenes names
    - **`rec_name`**: Name of the respective experiment from step 1.
    - **`exp_name`**: Name of the experiment
    - **`writer_name`**: Identifier for the logging writer
    - **`save_path`**: Directory where results will be saved
    - **`dataset_config.data_path`**: Path to the dataset created in the 'Data' step
    

     
    



     


    
