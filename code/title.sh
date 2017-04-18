#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=80000
#SBATCH --account=mccallum

n_components=100
n_iter=10000

python title.py --n_components=$n_components --n_iter=$n_iter
exit

