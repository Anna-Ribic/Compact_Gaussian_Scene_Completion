#!/bin/bash

# Hardcoded variables
logdir="your_logdir"           # Replace with your desired log directory
data="your_data"               # Replace with your data directory or scene
gpu=-1                         # Set to -1 for CPU; adjust as necessary for GPU
vsize=0.05                     # Voxel size
update_init_factor=16          # Update initialization factor
appearance_dim=0               # Appearance dimension
ratio=1                        # Ratio
lod=0                          # Level of detail (LOD)
iterations=10                  # Number of iterations
port_min=10000                 # Minimum port range for random port
port_max=30000                 # Maximum port range for random port
exp_name="test"
writer_name="test"
opt_file="bedrooms.txt"

# Function to generate a random port number within a range
function rand() {
    min=$1
    max=$(($2 - $min + 1))
    num=$(date +%s%N)
    echo $(($num % $max + $min))  
}

# Generate a random port number
port=$(rand $port_min $port_max)

# Generate timestamp for the output folder
time=$(date "+%Y-%m-%d_%H:%M:%S")

# Execute train_general.py with hardcoded parameters
python train_general.py --eval \
    -s data/${data} \
    --lod ${lod} \
    --gpu ${gpu} \
    --voxel_size ${vsize} \
    --update_init_factor ${update_init_factor} \
    --appearance_dim ${appearance_dim} \
    --ratio ${ratio} \
    --iterations ${iterations} \
    --port ${port} \
    -m outputs2/${data}/${logdir}/${time}\
    --exp_name ${exp_name} --writer_name ${writer_name} --opt_file ${opt_file}
