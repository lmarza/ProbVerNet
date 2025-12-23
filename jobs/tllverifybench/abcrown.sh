#!/usr/bin/env bash 
#SBATCH --job-name abcrown
#SBATCH --mem 64G
#SBATCH --cpus-per-gpu 24
#SBATCH --gres gpu:1
#SBATCH --partition normal
#SBATCH --output=tllVerifyBench_output.txt
#SBATCH --error=tllVerifyBench_error.txt

source /home/lmarza/miniconda3/etc/profile.d/conda.sh
conda activate alpha-beta-crown
export VNNCOMP_PYTHON_PATH=/home/lmarza/miniconda3/envs/alpha-beta-crown/bin
cd /home/lmarza/alpha-beta-CROWN/complete_verifier/
python abcrown.py --config exp_configs/vnncomp23/tllVerifyBench.yaml 

