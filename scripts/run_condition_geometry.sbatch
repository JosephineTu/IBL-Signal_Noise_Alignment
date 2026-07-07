#!/bin/bash
#SBATCH --job-name=cond_geom
#SBATCH --account=pi-bdoiron
#SBATCH --partition=caslake
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/cond_geom_%j.out
#SBATCH --error=logs/cond_geom_%j.err

source ~/.bashrc
source activate /scratch/midway3/xiaorantu/conda_envs/ibl

cd /home/xiaorantu/code/signal_noise_alignment

mkdir -p logs
python run_condition_geometry.py