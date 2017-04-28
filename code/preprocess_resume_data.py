import os, re, sys, argparse, multiprocess, timeit

import numpy as np
import cPickle as p

from operator import itemgetter
from xml.etree import ElementTree
from itertools import ifilterfalse
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
	regex = re.compile('[^a-zA-Z ]')

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


def get_title_sequence_data(n_resume_files, n_resume_files_skip=0, id_mapping=None):
	'''
	Reads the dataset of title sequences off disk, or creates it if it doesn't yet exist.
	'''
	# get a list of the files we are looking to parse for title sequences
	files = [ file for file in os.listdir('../data/') if 'resumes.xml' in file ][n_resume_files_skip:n_resume_files_skip + n_resume_files]

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

	low_frequency_titles = []

	print '\nFinding low-frequency job titles.'

	print '\nNumber of job titles before finding low-frequency ones:', len(set(titles))

	for idx, title in enumerate(sorted(set(titles))):
		if counts[title] < 10:
			low_frequency_titles.append(title)

	# for key in counts.keys():
	# 	if counts[key] < 10:
	# 		del counts[key]

	print '\nNumber of job titles after finding low-frequency ones:', len(set(titles)) - len(low_frequency_titles)

	start_time = timeit.default_timer()
	print '\nRemoving low-frequency job titles.'

	print '\nNumber of job sequences before removing low-frequency job titles:', len(data)
	
	# data[:] = ifilterfalse(lambda datum : any([ title in low_frequency_titles for title in datum ]), data)

	# print '\nNumber of job sequences after removing low-frequency job titles:', len(data)

	# print '\nIt took', timeit.default_timer() - start_time, 'seconds to remove low-frequency job titles.'
	# print '\nMapping job titles to unique integer IDs.'

	if id_mapping == None:
		id_mapping = {}

		for key in counts.keys():
			if counts[key] < 10:
				id_mapping[key] = 0

		start_time = timeit.default_timer()
		current_id = 1
		for title in set([ datum for l in data for datum in l ]).difference(set(id_mapping.keys())):
			if title not in id_mapping.keys():
				id_mapping[title] = current_id
				current_id += 1

	if id_mapping != None:
		for key in counts.keys():
			if key not in id_mapping.keys():
				id_mapping[key] = 0

	print '\nIt took', timeit.default_timer() - start_time, 'seconds to map job titles to unique integer IDs.'

	lengths = [ len(datum) for datum in data ]

	data = np.concatenate([ [ id_mapping[datum] for datum in l ]for l in data ]).reshape((-1, 1))

	return data, lengths, id_mapping, counts


if __name__ == '__main__':
	# create argument parser object
	parser = argparse.ArgumentParser(description='Pre-process resume files for various model-fitting.')

	parser.add_argument('--n_resume_files', type=int, default=15)

	# parse the arguments from the command line
	args = parser.parse_args()

	# get the parsed arguments
	n_resume_files = args.n_resume_files

	# get the sequential job title data
	print '\n...Importing sequential job title data.\n'
	if not 'sequential_title_data_' + str(n_resume_files) + '.p' in os.listdir('../data/'):
		data, lengths, mapping, counts = get_title_sequence_data(n_resume_files)
		p.dump((data, lengths, mapping, counts), open('../data/sequential_title_data_' + str(n_resume_files) + '.p', 'wb'))
	else:
		data, lengths, mapping, counts = p.load(open('../data/sequential_title_data_' + str(n_resume_files) + '.p', 'rb'))