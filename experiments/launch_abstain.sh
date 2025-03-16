#!/usr/bin/env bash
#SBATCH -p t4v2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH -a 0
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/abstain/slurm-%A_%a.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/abstain/slurm-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --exclude=gpu138


source /pkgs/anaconda3/bin/activate noisyTS

python3 -u run_abstain.py  --noise_type $1 --model_type $2 --dataset $3

