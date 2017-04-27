'''
Using a hidden Markov model to learn a transition model between job titles.

@author: Dan Saunders (djsaunde.github.io)
'''

import os, re, sys, argparse, multiprocess, timeit

import numpy as np
import cPickle as p

from operator import itemgetter
from xml.etree import ElementTree
from itertools import ifilterfalse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from hmmlearn.hmm import MultinomialHMM

from preprocess_resume_data import get_title_sequence_data, get_single_file_data


if __name__ == '__main__':
	# create argument parser object
	parser = argparse.ArgumentParser(description='Learn a model of job title trajectories using a multinomial hidden Markov model.')

	# add argument options for the number of components of the HMM and for the number of training iterations to use
	parser.add_argument('--n_components', type=int, default=10)
	parser.add_argument('--n_iter', type=int, default=100)
	parser.add_argument('--n_resume_files', type=int, default=15)

	# parse the arguments from the command line
	args = parser.parse_args()

	# get the parsed arguments
	n_components, n_iter, n_resume_files = args.n_components, args.n_iter, args.n_resume_files

	# get the sequential job title data
	print '\n...Importing sequential job title data.\n'
	if not 'sequential_title_data_' + str(n_resume_files) + '.p' in os.listdir('../data/'):
		data, lengths, mapping, counts = get_title_sequence_data(n_resume_files)
		p.dump((data, lengths, mapping, counts), open('../data/sequential_title_data_' + str(n_resume_files) + '.p', 'wb'))
	else:
		data, lengths, mapping, counts = p.load(open('../data/sequential_title_data_' + str(n_resume_files) + '.p', 'rb'))

	mapping = { value : key for key, value in mapping.iteritems() }

	print '\nNumber of distinct job titles (after lowercasing and removing non-letter characters:', len(set(mapping.values()))

	print '\nThere are', len(lengths), 'job title sequence examples.'

	print '\nListing job titles by ID\n'
	for key in sorted(mapping.keys()):
		print key, ':', mapping[key]

	print '\nListing job title counts by job title\n'
	for key, value in sorted(counts.items(), key=itemgetter(1)):
		print key, ':', counts[key]

	print '\n...Fitting multinomial hidden Markov model to job title sequence data.\n'

	# build and fit the hidden Markov model
	model = MultinomialHMM(n_components=n_components, n_iter=n_iter, verbose=True)
	model.fit(data, lengths)

	p.dump((model, mapping), open('../data/hmm_title_' + str(n_components) + '_' + str(n_resume_files) + '.p', 'wb'))