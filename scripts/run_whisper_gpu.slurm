#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -J whisper-lora-lt
#SBATCH -o /scratch/lustre/home/%u/whisper-ft/logs/slurm-%j.out
#SBATCH -e /scratch/lustre/home/%u/whisper-ft/logs/slurm-%j.err

set -e
cd /scratch/lustre/home/$USER/whisper-ft

mkdir -p logs
source venv/bin/activate

export HF_HOME=/scratch/lustre/home/$USER/whisper-ft/.cache/hf
export TRANSFORMERS_CACHE=/scratch/lustre/home/$USER/whisper-ft/.cache/hf
export HF_DATASETS_CACHE=/scratch/lustre/home/$USER/whisper-ft/.cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

python code/train_whisper_lora.py