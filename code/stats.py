'''
Script for calculating resume dataset statistics. More calculations will be added to this script as needed.

TODO: There are only first-order statistics about the number of job descriptions. We should add more statistics
as discussed in our weekly meetings.
'''

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cPickle as p
import numpy as np
import os


def jobs_per_resume():
    '''
    Returns the number of jobs per resume in the dataset.
    '''
    # counts number of job descriptions per resume
    description_counts = []

    # if we don't yet have the raw XML dataset stored on disk, get it
    if not 'job_counts.p' in os.listdir('../data/'):
        directories = os.listdir('../data')

        # look at all files in the 
        for file_name in directories:
            print '....Loading', file_name

            if '.xml' in file_name:
                tree = ET.parse('../data/' + file_name)
                root = tree.getroot()
                all_resume_tags = root.findall('.//resume')
            
                for resume_tag in all_resume_tags:
                	# add the number of job descriptions from this resume to the count
                    description_counts.append(len([ elem.tag for elem in resume_tag.iter() if elem.tag == 'Description' ]))

        # cast descriptions count to numpy array
        description_counts = np.array(description_counts)
        # save it to disk for next time
        p.dump(description_counts, open('../data/job_counts.p', 'wb'))

    # otherwise, we can simply load the XML dataset from disk
    else:
	    description_counts = p.load(open('../data/job_counts.p', 'rb'))

    # return the job description counts
    return description_counts


def compute_statistics(count_array, threshold):
    '''
    Calculates mean, median, number of resumes which have more than 'threshold' job description,
    and number of resumes. If threshold is >0(just to check if we are computing for job descriptions,
    as having a threshold doesn't make sense for most other attributes of the XMLs),return mean and median
    values.
    '''
    if threshold > 0:
        jobs_greater_than_threshold = len([ element for element in description_counts if element > threshold ])
        return np.mean(description_counts), np.median(description_counts), jobs_greater_than_threshold, len(description_counts)
    else:
        return np.mean(count_array),np.median(count_array)

def plot_histogram(count_array,feature):
    '''
    Plot a histogram of the number of the feature per resume.
    '''
    plt.figure(figsize=(16, 9))
    plt.hist(count_array, bins=range(max(count_array) + 1), rwidth=0.9)
    plt.title('Histogram of '+ feature+ ' per Resume')
    plt.xlabel(feature+' value')
    plt.ylabel('Number of resumes')
    plt.xticks(np.arange(0, max(count_array) + 1))
    
    plt.savefig('../plots/%s_per_resume_histogram.png',feature)
    
    plt.show()

def salaries_per_resume():
    '''
    Returns the salary in each resume in the dataset.
    '''
    #Salaries in each resume
    salaries = []

    # if we don't yet have the raw XML dataset stored on disk, get it
    if not 'salaries.p' in os.listdir('../data/'):
        directories = os.listdir('../data')

        # look at all files in the data directory
        for file_name in directories:
            print '....Loading', file_name

            if '.xml' in file_name:
                tree = ET.parse('../data/' + file_name)
                root = tree.getroot()
                all_resume_tags = root.findall('.//resume')
            
                for resume_tag in all_resume_tags:
                    for elem in resume_tag.iter():
                        if elem.tag == 'Salary':
                            salaries.append(int(float(elem.text)))
                	# add the number of job descriptions from this resume to the count
                    #salaries.append(int(elem.text for elem in resume_tag.iter() if elem.tag == 'Salary'))

        # cast descriptions count to numpy array
        salaries = np.array(salaries)
        # save it to disk for next time
        p.dump(salaries, open('../data/salaries.p', 'wb'))

    # otherwise, we can simply load the XML dataset from disk
    else:
	    salaries = p.load(open('../data/salaries.p', 'rb'))

    # return the job description counts
    return salaries



if __name__ == '__main__':
    # set threshold for 'jobs_greater_than_threshold' calculation in 'compute_statistics()'
    threshold = 1

    # get job description counts per resume
    #description_counts = jobs_per_resume()
    
    #print max(description_counts), min(description_counts)

    # plot a histogram of job description counts
    #plot_jobs_per_resume(description_counts)

    # calculate count statistics for job descriptions per resume
    #stats = compute_statistics(description_counts, threshold)
    
    # print job description counts statistics
    #print '\nMean number of resumes:', stats[0], '\nMedian number of resumes:', stats[1], '\nNumber of resumes with more than', \
    #threshold, 'job:', stats[2], '\nNumber of resumes in corpus:', stats[3], '\n'
    
    salaries = salaries_per_resume()
    print "Computing stats...."
    stats = compute_statistics(salaries,0)
    
    print '\nMean salary:', stats[0], '\nMedian salary:', stats[1]
    
    print "\nPlotting histogram..."
    plot_histogram(salaries,"Salaries")
    
