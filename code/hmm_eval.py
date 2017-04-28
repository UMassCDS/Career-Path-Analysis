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
	parser = argparse.ArgumentParser(description='Evaluate saved HMM model on held-out test data.')

	# add argument options for the number of components of the HMM
	parser.add_argument('--n_components', type=int, default=10)
	parser.add_argument('--n_resume_files', type=int, default=15)
	parser.add_argument('--n_resume_files_skip', type=int, default=15)
	
	args = parser.parse_args()
	n_components, n_resume_files, n_resume_files_skip = args.n_components, args.n_resume_files, args.n_resume_files_skip

	# get old mapping
	print '../data/sequential_title_data_' + str(n_resume_files_skip) + '.p'
	f = open('../data/sequential_title_data_' + str(n_resume_files_skip) + '.p')
	print f
	_, _, mapping, _ = p.load(open('../data/sequential_title_data_' + str(n_resume_files_skip) + '.p', 'rb'))

	# get the sequential job title data
	print '\n...Importing sequential job title data.\n'
	if not 'sequential_title_data_test_' + str(n_resume_files) + '.p' in os.listdir('../data/'):
		data, lengths, mapping, counts = get_title_sequence_data(n_resume_files, n_resume_files_skip, mapping)
		p.dump((data, lengths, mapping, counts), open('../data/sequential_title_data_test_' + str(n_resume_files) + '.p', 'wb'))
	else:
		data, lengths, mapping, counts = p.load(open('../data/sequential_title_data_test_' + str(n_resume_files) + '.p', 'rb'))

	mapping = { value : key for key, value in mapping.iteritems() }

	print '\nNumber of distinct job titles (after lowercasing and removing non-letter characters:', len(set(mapping.values()))
	print '\nThere are', len(lengths), 'job title sequence examples.'

	model, mapping = p.load(open('../data/hmm_title_' + str(n_components) + '_' + str(n_resume_files_skip) + '.p', 'rb'))

	print '\nLog-likelihood of data under HMM model:', model.score(data)