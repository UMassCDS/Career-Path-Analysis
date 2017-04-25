'''
Using a hidden Markov model to learn a transition model between job titles.

@author: Dan Saunders (djsaunde.github.io)
'''

import os, re, sys, argparse, multiprocess, timeit

import numpy as np
import cPickle as p

from operator import itemgetter
from xml.etree import ElementTree
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from hmmlearn.hmm import MultinomialHMM


def get_single_file_data(file, file_index, num_files):
	'''
	Reads in a single file of resume data (contains 50,000 resumes).
	'''
	print '...Loading resume file', file_index + 1, '/', num_files

	# create data structure to store job title sequences
	title_sequences = []
	counts = {}

	# for removing non-letter characters
	regex = re.compile('[A-Z ]{6,}[A-Z ]{6,}')

	# parse the XML tree
	tree = ElementTree.parse('../data/' + file)
	root = tree.getroot()

	# getting all resumes
	all_resume_tags = root.findall('.//resume')

	# For each resume in each file
	for resume_tag in all_resume_tags:
		jobs = resume_tag.findall('.//experiencerecord')

		# If there are more than two transitions in that resume
		if len(jobs) >= 2:
			job_list = []
			start_year = 0
			title = None

			# For each transition in that resume
			for job in jobs:
				for tag in job:
					if tag.tag == 'Title':
						if tag.text is not None and tag.text != '':
							title = regex.sub('', tag.text.lower())
							if title not in counts.keys():
								counts[title] = 1
							else:
								counts[title] += 1

					if tag.tag == 'StartYear':
						start_year = int(tag.text)

				# Make a tuple of the start year and description
				year_title_tuple = (start_year, title)

				# Append the tuple to list of transitions in that resume
				job_list.append(year_title_tuple)

			# Sort them by year to create the sequence
			job_list = sorted(job_list, key=itemgetter(0))

			# Append the sequence from that resume to the master list
			if len(job_list) >= 2:
				to_append = []
				for job in job_list:
					if job[1] != None and job[1] != '':
						to_append.append(job[1])
				if len(to_append) >= 2:
					title_sequences.append(to_append)

	return title_sequences, counts


def get_title_sequence_data():
	'''
	Reads the dataset of title sequences off disk, or creates it if it doesn't yet exist.
	'''
	# get a list of the files we are looking to parse for title sequences
	files = [ file for file in os.listdir('../data/') if 'resumes.xml' in file ][:50]

	start_time = timeit.default_timer()
	out = Parallel(cpu_count())(delayed(get_single_file_data)(file, idx, len(files)) for idx, file in enumerate(files))
	
	title_sequences = []
	counts = {}
	for output in out:
		title_sequences.extend(output[0])
		counts = dict((k, counts.get(k, 0) + output[1].get(k, 0)) for k in set(counts.keys()) | set(output[1].keys()))
	print '\nIt took', timeit.default_timer() - start_time, 'seconds to load the resume data.'

	data = title_sequences
	titles = [ title for datum in data for title in datum ]

	id_mapping = {}

	for idx, title in enumerate(sorted(set(titles))):
		if counts[title] < 10:
			id_mapping[title] = 0


	start_time = timeit.default_timer()
	print '\nRemoving low-frequency job titles.\n'
	
	for idx, datum in enumerate(data):
		if idx % 500 == 0:
			print 'Progress:', idx, '/', len(data)
		for title in datum:
			if title in id_mapping.keys():
				data.remove(datum)
				break

	print 'Progress:', len(data), '/', len(data), '\n'
	print '\nIt took', timeit.default_timer() - start_time, 'seconds to remove low-frequency job titles.'

	print '\nMapping job titles to unique integer IDs.'

	start_time = timeit.default_timer()
	current_id = 0
	for title in set([ datum for l in data for datum in l ]).difference(set(id_mapping.keys())):
		if title not in id_mapping.keys():
			id_mapping[title] = current_id
			current_id += 1

	print '\nIt took', timeit.default_timer() - start_time, 'seconds to map job titles to unique integer IDs.'

	lengths = [ len(datum) for datum in data ]
	data = np.concatenate([ np.array([ id_mapping[datum] for datum in l ]) for l in data ]).reshape((-1, 1))

	return data, lengths, id_mapping


if __name__ == '__main__':
	# create argument parser object
	parser = argparse.ArgumentParser(description='Learn a model of job title trajectories using a multinomial hidden Markov model.')

	# add argument options for the number of components of the HMM and for the number of training iterations to use
	parser.add_argument('--n_components', type=int, default=10)
	parser.add_argument('--n_iter', type=int, default=100)

	# parse the arguments from the command line
	args = parser.parse_args()

	# get the parsed arguments
	n_components, n_iter = args.n_components, args.n_iter

	# get the sequential job title data
	print '\n...Importing sequential job title data.\n'
	if not 'sequential_title_data.p' in os.listdir('../data/'):
		data, lengths, mapping = get_title_sequence_data()
		p.dump((data, lengths, mapping), open('../data/sequential_title_data.p', 'wb'))
	else:
		data, lengths, mapping = p.load(open('../data/sequential_title_data.p', 'rb'))

	print '\nNumber of distinct job titles (after lowercasing and removing non-letter characters:', len(set(mapping.values()))

	print '\nThere are', data.shape[0], 'job title sequence examples.'

	print '\n...Fitting multinomial hidden Markov model to job title sequence data.\n'

	# build and fit the hidden Markov model
	model = MultinomialHMM(n_components=n_components, n_iter=n_iter, verbose=True)
	model.fit(data, lengths)
