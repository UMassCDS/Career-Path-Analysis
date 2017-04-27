import os, re, sys, argparse, multiprocess, timeit

import numpy as np
import cPickle as p

from operator import itemgetter
from xml.etree import ElementTree
from itertools import ifilterfalse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from hmmlearn.hmm import MultinomialHMM


if __name__ == '__main__':
	# create argument parser object
	parser = argparse.ArgumentParser(description='Learn a model of job title trajectories using a multinomial hidden Markov model.')

	# add argument options for the number of components of the HMM and for the number of training iterations to use
	parser.add_argument('--n_components', type=int, default=5)
	parser.add_argument('--n_resume_files', type=int, default=5)

	# parse the arguments from the command line
	args = parser.parse_args()

	# get the parsed arguments
	n_components, n_resume_files = args.n_components, args.n_resume_files

	# get the sequential job title data
	model, mapping = p.load(open('../data/hmm_title_' + str(n_components) + '_' + str(n_resume_files) + '.p', 'rb'))

	for idx in xrange(15):
		sample = model.sample(n_samples=10)
		print [ mapping[arr[0]] for arr in sample[0] ]
		print model.score(sample[0])