#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -J whisper-eval-lt
#SBATCH -o /scratch/lustre/home/%u/whisper-ft/logs/eval-%j.out
#SBATCH -e /scratch/lustre/home/%u/whisper-ft/logs/eval-%j.err

set -e
cd /scratch/lustre/home/$USER/whisper-ft

mkdir -p logs
source venv/bin/activate

export HF_HOME=/scratch/lustre/home/$USER/whisper-ft/.cache/hf
export TRANSFORMERS_CACHE=/scratch/lustre/home/$USER/whisper-ft/.cache/hf
export HF_DATASETS_CACHE=/scratch/lustre/home/$USER/whisper-ft/.cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

python3 code/eval_whisper_lora_only.py