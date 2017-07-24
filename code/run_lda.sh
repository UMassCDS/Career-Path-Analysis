#!/bin/bash
#
#SBATCH --partition=defq
#SBATCH --time=0-06:00:00
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=8

echo "hello"
date

cd /home/rattigan/careerpath/Career-Path-Analysis
source env/bin/activate

cd code
time python lda.py /home/rattigan/careerpath/Career-Path-Analysis/data/timelines_med.p 400 20

echo "goodbye"
date

exit
