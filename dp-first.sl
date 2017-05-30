#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 00:04:00
#SBATCH -C haswell
#SBATCH -o dp-first.txt

srun -n 2 python ./deep-net.py
