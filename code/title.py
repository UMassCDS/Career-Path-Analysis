'''
Replicating the TITLE model from "Modeling Career Path Trajectories" (Mimno & McCallum 2008, unpublished).

@author: Dan Saunders (djsaunde.github.io)
'''

import os, re, sys, argparse

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

	# for removing non-letter characters
	regex = re.compile( "[A-Z ]{6,}[A-Z ]{6,}")

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

	return title_sequences



def get_title_sequence_data():
	'''
	Reads the dataset of title sequences off disk, or creates it if it doesn't yet exist.
	'''
	# get a list of the files we are looking to parse for title sequences
	files = [ file for file in os.listdir('../data/') if 'resumes.xml' in file ]

	title_sequences = Parallel(cpu_count())((delayed(get_single_file_data)(file, idx, len(files)) for idx, file in enumerate(files)))

	return title_sequences


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
		data = get_title_sequence_data()
		p.dump(data, open('../data/sequential_title_data.p', 'wb'))
	else:
		data = p.load(open('../data/sequential_title_data.p', 'rb'))

	data = [ datum for l in data for datum in l ][:5000]

	print '\nNumber of distinct job titles (after lowercasing and removing non-letter characters:', len(set([ datum for l in data for datum in l ]))

	# for idx, title in enumerate(set([ datum for l in data for datum in l ])):
	# 	print idx	
	# 	if ([ datum for l in data for datum in l ]).count(title) < 10:
	# 		data = [ 'X' if datum == title else datum for l in data for datum in l ]

	id_mapping = {}
	current_id = 0
	for title in set([ datum for l in data for datum in l ]):
		id_mapping[title] = current_id
		current_id += 1

	lengths = [ len(datum) for datum in data ]
	data = np.concatenate([ np.array([ id_mapping[datum] for datum in l ]) for l in data ]).reshape((-1, 1))

	print '\nThere are', data.shape[0], 'job title sequence examples.'

	print '\n...Fitting multinomial hidden Markov model to job title sequence data.\n'

	# build and fit the hidden Markov model
	model = MultinomialHMM(n_components=n_components, n_iter=n_iter, verbose=True)
	model.fit(data, lengths)
