import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import glob
import os

def xmlToDataFrame():
    listOfDirs=os.listdir("data")
    #print listOfDirs
    i=0
    headers=[]
    for filename in listOfDirs:
        #fileName = filename.split('.')[0]+'.'+filename.split('.')[1]
        print filename
        if "xml" in filename:
            tree = ET.parse('data/'+filename)
            root = tree.getroot()
            allRecords =[]
            jobDesc = ''
            jobTitle =''
            allJobDesc = []
            #allJobTitles = []
            jobDescTitle = []
            allDescTags = root.findall('.//Description')
            allTitleTags = root.findall('.//Title')
            for tagDesc in allDescTags:
                if tagDesc.text is not None:
                    allJobDesc.append(tagDesc.text)  
            #for tagTitle in allTitleTags:
             #   allJobTitles.append(tagTitle.text)
    
    return allJobDesc

    #print allJobTitles    
#print df.dtypes
#print df.shape
