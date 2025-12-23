#!/usr/bin/env bash 
#SBATCH --job-name abcrown
#SBATCH --mem 64G
#SBATCH --cpus-per-gpu 24
#SBATCH --gres gpu:1
#SBATCH --partition normal
#SBATCH --output=acas_output.txt
#SBATCH --error=acas_error.txt

source /home/lmarza/miniconda3/etc/profile.d/conda.sh
conda activate prob-ver
export VNNCOMP_PYTHON_PATH=/home/lmarza/miniconda3/envs/alpha-beta-crown/bin
cd /home/lmarza/alpha-beta-CROWN/complete_verifier/
python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml

