#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=40000
#SBATCH --account=mccallum

n_components=10
n_iters=10000

python title.py --n_components=$n_components --n_iters=$n_iters
exit

