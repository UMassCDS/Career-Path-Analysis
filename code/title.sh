#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=80000
#SBATCH --account=mccallum

n_components=100
n_iter=10000
n_resume_files=5

python hmm_title.py --n_components=$n_components --n_iter=$n_iter --n_resume_files=$n_resume_files
exit

