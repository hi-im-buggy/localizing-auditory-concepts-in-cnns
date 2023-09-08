#!/bin/bash
#SBATCH --job-name=storing-activations
#SBATCH -A research
#SBATCH -p long
#SBATCH -w gnode015
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=3000
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratyaksh.g@research.iiit.ac.in
#SBATCH --output=/tmp/jupyter-slurm-log
#SBATCH --time=4-00:00:00

source /home2/pratyaksh.g/.bashrc
conda activate ms-clap

cd /home2/pratyaksh.g/MS-CLAP/src
python3 experiments/0-storing-activations.py
