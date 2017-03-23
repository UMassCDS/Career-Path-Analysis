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
			
			# Getting all resumes..."
			allResumeTags = root.findall('.//resume')
				
			# collect sequences of job description text
			allJobDesc = []
			
			#For each resume in each file
                        for resumeTag in allResumeTags:
                            expTagsInResume = resumeTag.findall('.//experiencerecord')
                            
			    #If there are more than two transitions in that resume
			    if len(expTagsInResume)>=2:
                                jobTupleList = []
				startYear = 0
				filteredDesc = None
				sortedByYearTuples = []
                                
				#For each transition in that resume
				for expTag in expTagsInResume:
                                    for memberTag in expTag.iter():
                                        if memberTag.tag == 'Description':
				            if memberTag.text is not None:
					        filteredDesc = str([ word.lower() for word in memberTag.text.split() if word.lower() not in stop and word.isalpha() ])
                                        if memberTag.tag == 'StartYear':
                                            startYear = int(memberTag.text)
                                    	
					#Make a tuple of the start year and description
					jobYearTuple = (startYear,filteredDesc)

                                    #Append the tuple to list of transitions in that resume
				    jobTupleList.append(jobYearTuple)

                                    #Sort them by year to create the sequence
				    sortedByYearTuples = sorted(jobTupleList,key = itemgetter(0))
                                
				#Append the sequence from that resume to the master list
				allJobDesc.append(sortedByYearTuples)
            
					      
	
	# pickle the job description sequences
	p.dump(allJobDesc, open('../data/resume_data.p', 'wb'))

if __name__ == '__main__':
	buildDataset()
