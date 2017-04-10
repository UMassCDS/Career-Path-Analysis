'''
Parses XML dataset into sequences of job descriptions.

TODO: We need to parse corresponding job titles and perhaps supporting metadata from the resumes.
'''

import xml.etree.ElementTree as ET
import pandas as pd
import os
import cPickle as p
from operator import itemgetter
from nltk.corpus import stopwords
from operator import itemgetter


def build_dataset():
    '''
    Collects job description sequences for all the resumes in the dataset
    '''
    # get the names of all files in the data directory
    list_of_dirs = os.listdir('../data')
    stop = set(stopwords.words('english'))

    all_job_desc = []

    # look through all files in the data directory
    for idx, filename in enumerate(list_of_dirs):
	print '...Loading', filename, '(',  str(idx + 1), '/', str(len(list_of_dirs)), ')' 

	# if the file is an XML file...
	if '.xml' in filename:
	    # parse the XML tree
	    tree = ET.parse('../data/' + filename)
	    root = tree.getroot()

	    # Getting all resumes..."
	    all_resume_tags = root.findall('.//resume')

	    # For each resume in each file
	    for resume_tag in all_resume_tags:
		exp_tags_in_resume = resume_tag.findall('.//experiencerecord')

		# If there are more than two transitions in that resume
		if len(exp_tags_in_resume) >= 2:
		    job_tuple_list = []
		    start_year = 0
		    filtered_desc = None
		    sorted_by_year_tuples = []

		    # For each transition in that resume
		    for exp_tag in exp_tags_in_resume:
			for member_tag in exp_tag.iter():
			    if member_tag.tag == 'Description':
				if member_tag.text is not None:
				    filtered_desc = [ word.lower() for word in member_tag.text.split() if word.lower() not in stop and word.isalpha() ]
				    if member_tag.tag == 'StartYear':
				        start_year = int(member_tag.text)

				    # Make a tuple of the start year and description
				    job_year_tuple = (start_year, filtered_desc)

				    # Append the tuple to list of transitions in that resume
				    job_tuple_list.append(job_year_tuple)

				    # Sort them by year to create the sequence
				    sorted_by_year_tuples = sorted(job_tuple_list, key=itemgetter(0))

				    # Append the sequence from that resume to the master list
				    if len(sorted_by_year_tuples) >= 2:
					to_append = []
					for job in sorted_by_year_tuples:
					    if job[1] != None:
					        to_append.append(job[1])
					    if len(to_append) >= 2:
						all_job_desc.append(to_append)
		
        print '...we\'ve parsed', len(all_job_desc), 'job descriptions so far'
		
	# pickle the job description sequences
    p.dump(all_job_desc, open('../data/resume_data.p', 'wb'))


if __name__ == '__main__':
    build_dataset()
