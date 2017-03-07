import xml.etree.ElementTree as ET
import pandas as pd
import os
import cPickle as p
from nltk.corpus import stopwords

def xmlToDataset():
	'''
	Converts an XML resume file into a pandas DataFrame to use with scikit-learn's
	LDA implementation.
	'''
	# get the names of all files in the data directory
	listOfDirs = os.listdir('../data')
	stop = set(stopwords.words('english'))
	
	# look through all files in the data directory
	for idx, filename in enumerate(listOfDirs):
		print '...Loading', filename, '(',  str(idx + 1), '/', str(len(listOfDirs)), ')' 
		
		# if the file is an XML file...
		if 'xml' in filename:
			# parse the XML tree
			tree = ET.parse('../data/' + filename)
			root = tree.getroot()
			
			# get all tags having to do with job descriptions
			allDescTags = root.findall('.//Description')
				
			# collect the job description text from the XML
			allJobDesc = []
			for tagDesc in allDescTags:
				if tagDesc.text is not None:
					filteredDesc = [ word.lower() for word in tagDesc.text.split() if word.lower() not in stop ]
					allJobDesc.append(str(filteredDesc))  
	
	# pickle the job description text
	p.dump(allJobDesc, open('../data/resume_data.p', 'wb'))

if __name__ == '__main__':
	xmlToDataset()
