#!/usr/bin/env bash
#SBATCH -p t4v2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH -a 0
#SBATCH --qos=m
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/regret/slurm-%A.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/regret/slurm-%A.out
#SBATCH --open-mode=append
#SBATCH --exclude=gpu138

source /pkgs/anaconda3/bin/activate noisyTS

python3 -u run_regret.py  --noise_type $1 --model_type $2 --dataset $3 --noise_level $4