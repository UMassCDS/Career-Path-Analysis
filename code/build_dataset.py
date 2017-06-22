"""
Parses XML dataset into sequences of job descriptions.

TODO: We need to parse corresponding job titles and perhaps supporting metadata from the resumes.
"""

import sys
import xml.etree.ElementTree as ET
import cPickle as pickle
from nltk.corpus import stopwords
from operator import itemgetter

    
def build_dataset(infilenames, outfile):
    """
    Collects job description sequences for all the resumes in the dataset
    """
    # get the names of all files in the data directory
    stop = set(stopwords.words('english'))

    all_job_desc = []

    # look through all files in the data directory
    for idx, filename in infilenames:
        print "...Loading {} ({}/{})".format(filename, idx + 1, len(infilenames))

        # if the file is an XML file...
        if '.xml' in filename:
            # parse the XML tree
            tree = ET.parse('../data/' + filename)
            root = tree.getroot()

            # Getting all resumes...
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
                                    filtered_desc = [ w.lower() for w in member_tag.text.split()
                                                        if w.lower() not in stop and w.isalpha() ]
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
    with open(outfile, 'wb') as outp:
        pickle.dump(all_job_desc, outp)

    return len(all_job_desc)

if __name__ == '__main__':
    ins = sys.argv[1:-1]
    out = sys.argv[-1]
    build_dataset(ins, out)

