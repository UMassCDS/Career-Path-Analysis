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


def compute_statistics(description_counts, threshold):
    '''
    Calculates mean, median, number of resumes which have more than 'threshold' job description,
    and number of resumes.
    '''
    jobs_greater_than_threshold = len([ element for element in description_counts if element > threshold ])
    return np.mean(description_counts), np.median(description_counts), jobs_greater_than_threshold, len(description_counts)


def plot_jobs_per_resume(description_counts):
    '''
    Plot a histogram of the number of job descriptions per resume.
    '''
    plt.figure(figsize=(16, 9))
    plt.hist(description_counts, bins=range(max(description_counts) + 1), rwidth=0.9)
    plt.title('Histogram of Jobs per Resume')
    plt.xlabel('Number of jobs')
    plt.ylabel('Number of resumes')
    plt.xticks(np.arange(0, max(description_counts) + 1))
    
    plt.savefig('../plots/jobs_per_resume_histogram.png')
    
    plt.show()


if __name__ == '__main__':
    # set threshold for 'jobs_greater_than_threshold' calculation in 'compute_statistics()'
    threshold = 1

    # get job description counts per resume
    description_counts = jobs_per_resume()
    
    print max(description_counts), min(description_counts)

    # plot a histogram of job description counts
    plot_jobs_per_resume(description_counts)

    # calculate count statistics for job descriptions per resume
    stats = compute_statistics(description_counts, threshold)
    
    # print job description counts statistics
    print '\nMean number of resumes:', stats[0], '\nMedian number of resumes:', stats[1], '\nNumber of resumes with more than', \
    threshold, 'job:', stats[2], '\nNumber of resumes in corpus:', stats[3], '\n'