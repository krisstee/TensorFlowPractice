#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -C haswell
#SBATCH -o nn-out.txt

srun -n 4 python ./convolution_nn/cnn_mnist.py
