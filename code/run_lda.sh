#!/bin/bash
#
#SBATCH --partition=defq
#SBATCH --time=0-06:00:00
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=32


# args: data_file num_topics num_procs

DATAFILE=$1
NUMTOPICS=$2
NUMJOBS=$3

echo "hello"
date

cd /home/rattigan/careerpath/Career-Path-Analysis
source env/bin/activate

cd code
#time python lda.py /home/rattigan/careerpath/Career-Path-Analysis/data/timelines_sm.p 400 20
echo "python lda.py $1 $2 $3"
time python lda.py $1 $2 $3

echo "goodbye"
date

exit
