import xml.etree.ElementTree as ET
import pandas as pd
import os
import cPickle as p
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
def jobsPerResume():
    jobCount = []
    countOfDesc = []
    listOfDirs = os.listdir('../data')
    for filename in listOfDirs:
        print '....Loading',filename

        if 'xml' in filename:
            tree = ET.parse('../data/'+filename)
            root = tree.getroot()
            allResumeTags = root.findall('.//resume')
            for resumeTag in allResumeTags:
                countOfDesc.append(len( [elem.tag for elem in resumeTag.iter() if elem.tag =='Description']))
    return np.array(countOfDesc)

def computeStatistics(countOfDesc,threshold):

    jobsGreaterThanThreshold = len([element for element in countOfDesc if element>threshold])
    return [np.mean(countOfDesc),np.median(countOfDesc),jobsGreaterThanThreshold,len(countOfDesc)]

def plotJobsPerResume(countOfDes):    
    plt.hist(countOfDes,bins = 'auto')
    plt.xlabel('Number of jobs')
    plt.ylabel('Number of resumes')
    plt.xticks(np.arange(0,20))
    plt.show()
    
threshold=1
countList = jobsPerResume()
plotJobsPerResume(countList)
statsList = computeStatistics(countList,threshold)
print statsList
