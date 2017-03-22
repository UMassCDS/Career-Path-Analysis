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


def buildDataset():
	'''
	Collects job description sequences for all the resumes in the dataset
        TODO: Have tuples of (year,description) in each sequence, remove the year 
        stuff
        '''
	# get the names of all files in the data directory
	listOfDirs = os.listdir('../data')
	stop = set(stopwords.words('english'))
	
	# look through all files in the data directory
        for idx, filename in enumerate(listOfDirs):
		print '...Loading', filename, '(',  str(idx + 1), '/', str(len(listOfDirs)), ')' 
		
		# if the file is an XML file...
		if '.xml' in filename:
			# parse the XML tree
			tree = ET.parse('../data/' + filename)
			root = tree.getroot()
			
			# get all tags havLing to do with job descriptions
			allResumeTags = root.findall('.//resume')
				
			# collect sequences of job description text
			allJobDesc = []
                        for resumeTag in allResumeTags[0:2]:
                            expTagsInResume = resumeTag.findall('.//experiencerecord')
                            if len(expTagsInResume)>=2:
                                jobTupleList = []
                                for expTag in expTagsInResume:
                                    for memberTag in expTag.iter():
                                        if memberTag.tag == 'Description':
				            if memberTag.text is not None:
					        filteredDesc = str([ word.lower() for word in memberTag.text.split() if word.lower() not in stop and word.isalpha() ])
                                        if memberTag.tag == 'StartYear':
                                            startYear = int(memberTag.text)
                                            jobYearTuple = (startYear,filteredDesc)
                                    jobTupleList.append(jobYearTuple)
                                sortedByYearTuples = sorted(jobTupleList,key = itemgetter(0))
                            allJobDesc.append(sortedByYearTuples)
            
					      
	
	# pickle the job description sequences
	p.dump(allJobDesc, open('../data/resume_data.p', 'wb'))

if __name__ == '__main__':
	buildDataset()
