#!/bin/bash
#
#SBATCH --partition=defq
#SBATCH --time=0-06:00:00
#SBATCH --mem=128000

echo "hello"
date

cd /home/rattigan/careerpath/Career-Path-Analysis
source env/bin/activate
cd code
time python timelines.py $*

echo "goodbye"
date

exit
