
# General Scaffold-GS Reconstruction

This repository provides a training procedure for reconstruction multiple scenes with fixed attribute mlps using the Scaffold-GS approach 

## Data


## Reconstruction Pipeline
    
1. Run Scaffold-GS reconstruction on multiple scenes
   - **Usage**:
     ```bash
     bash train_general.sh
     ```
   - **Arguments**:
     - `--exp_name`: Name to where results and logging information is stored.
     - `--writer_name`: Tensorboard writer name
     - `--feat_only`: If attribute mlps should be fixed or not
     - `--voxel_size`: Voxel size of reconstruction
     - `--opt_file`: Path to scenes that should be reconstructed.


    