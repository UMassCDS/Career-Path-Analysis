import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import cPickle as p
import os
import operator

all_job_transitions = dict()

job_transitions_data = "job_transitions.p"
job_transitions_data_filepath = "../data/"

resume_data_directory = "../data/"


def update_dict(titles):
    i = 0
    while i < (len(titles) - 1):
        if titles[i] in all_job_transitions:
            di = all_job_transitions[titles[i]]
        else:
            di = dict()
        di[titles[i+1]] = di.get(titles[i+1],0) + 1 
        #print "T0 is ", titles[i], "T1 is", titles[i+1]
        all_job_transitions[titles[i]] = di
        i = i+1
 


def predict_next_job_title(current_title):
    if current_title in all_job_transitions:
       di = all_job_transitions[current_title]
       sorted_x = sorted(di.items(), key=operator.itemgetter(1), reverse=True)
       #print "sorted dictionary is ", sorted_x
       if len(sorted_x) > 3:
           print "NEXT JOB TITLEs ARE ", sorted_x[0][0]," ,", sorted_x[1][0], " ,", sorted_x[2][0]
       elif len(sorted_x) < 2:
           print "NEXT JOB TITLEs ARE ", sorted_x[0][0]
       else:
           print "NEXT JOB TITLEs ARE ", sorted_x[0][0], " ,", sorted_x[1][0]

    else:
       print "Cannot predict as we haven't seen this job title before"
       


def create_transitions(all_resume_tags):

    for resume_tag in all_resume_tags:
        for elem in resume_tag:
            sorted_titles = list()
            titles = list()
            start_year = list()
            
            #all_company_tags = resume_tag.findall('.//Company')
    
            if elem.tag == "experience":
                all_title_tags = elem.findall('.//Title')
                for t in all_title_tags:
                    #print "title is ", t.text
                    titles.append(t.text)
                    
                
                start_year_tags = elem.findall('.//StartYear')
    
                for sy in start_year_tags:
                    #print "start_year is ", sy.text
                    start_year.append(sy.text)
                
                sorted_titles = [x for (y,x) in sorted(zip(start_year,titles))]
                
                #print "sorted_titles!" , sorted_titles
                update_dict(sorted_titles)
        #print "rt is", resume_tag
        

def predict_for_all_job_titles():

    for k in all_job_transitions.keys():
        print "For job", k
        predict_next_job_title(k)
    

        


if __name__ == "__main__":


    if not job_transitions_data in os.listdir(job_transitions_data_filepath):
        directories = os.listdir(resume_data_directory)
        
        for file_name in directories:
            print '....Loading', file_name
        
            if '.xml' in file_name and file_name != "resumes.xml10000":
                tree = ET.parse(resume_data_directory + file_name)
                root = tree.getroot()
                all_resume_tags = root.findall('.//resume')

                create_transitions(all_resume_tags)
            
                p.dump(all_job_transitions, open(job_transitions_data_filepath + job_transitions_data, 'wb'))
    
    else:
        all_job_transitions = p.load(open(job_transitions_data_filepath + job_transitions_data, 'rb'))
        
    
    #predict_for_all_job_titles() 
        
    
    #predict_next_job_title("CEO")
