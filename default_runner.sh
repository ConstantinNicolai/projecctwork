#!/bin/bash
#SBATCH --partition=brook
#SBATCH --job-name=read_gpu_stats
#SBATCH --outputresnet101_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

# Check if Nvidia SMI is installed
python3 resnet101.py
