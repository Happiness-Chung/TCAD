#!/bin/bash
#SBATCH --job-name=TCAD_CheXpert_resnet_layer_depth_128_seed1       # Job name
#SBATCH --nodes=1                      # Total # of nodes
#SBATCH --nodelist=node3
#SBATCH --ntasks-per-node=1          # # of tasks per node
#SBATCH --cpus-per-task=16              # # of cores per task
#SBATCH --time=72:00:00                # Time limit (hh:mm:ss)
#SBATCH --output=/scratch/connectome/stellasybae/TCAD/slurm_logs/%A_%x.out         # Standard error
#SBATCH --error=/scratch/connectome/stellasybae/TCAD/slurm_logs/%A_%x.err          # Standard output
#SBATCH --mail-user=stellasybae@snu.ac.kr  # Email address
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --gpus-per-task=1


## 03 labserver version
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate divfreq

python main.py --experiment_name 240513_CheXpert_resnet_layer_depth_128_seed1 --model ResNet --seed 1 --layer_depth 128 --dataset CheXpert