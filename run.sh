#!/bin/bash
#SBATCH --job-name=replication
#SBATCH --cpus-per-task=1
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -e /home/sc10264/samplingrnn/code_planning_representation/results/slurm-%A_%a.err
#SBATCH -o /home/sc10264/samplingrnn/code_planning_representation/results/slurm-%A_%a.out
#SBATCH --array=0

python -u training.py --jobid=$SLURM_ARRAY_TASK_ID