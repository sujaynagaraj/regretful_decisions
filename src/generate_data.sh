#!/bin/bash
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/slurm-%A.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/slurm-%A.out
#SBATCH --qos=nopreemption


source /pkgs/anaconda3/bin/activate noisyTS
      
python3 -u generate_data.py 

