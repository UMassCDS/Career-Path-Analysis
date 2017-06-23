"""
Template for running LDA on job description corpus.

@author: Dan Saunders (djsaunde.github.io)
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from build_dataset import build_dataset

import sys
import matplotlib.pyplot as plt
import os
import timeit
import cPickle as p
import numpy as np


np.set_printoptions(threshold=np.nan)


def print_top_words(model, feature_names, n_words, n_topic=0):
    """
    Prints top 'n_words' words from the topics 'feature_names'.

    model: The LDA model from which to print representative words from topics.
    feature_names: The names of the representative word tokens.
    n_words: The number of representative words to print for each topic.
    """
    if n_topic == 0:
        for topic_idx, topic in enumerate(model.components_):
            print 'Topic #%d:' % topic_idx
            print ' '.join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
        print '\n'
    else:
        for topic_idx, topic in enumerate(model.components_):
            if topic_idx == n_topic:
                print 'Topic #%d:' % topic_idx
                print ' '.join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
     

def flatten_descrip_seqs(job_desc_seqs):
    data = []
    job_sequence_counts = []
    total_job_count = 0
    for datum_idx, datum in enumerate(job_desc_seqs):
        job_count = 0
        for job_descr_indx, job_description in enumerate(datum):
            data.append(' '.join(job_description))
            job_count += 1
            total_job_count += 1
        if job_count != 0:
            job_sequence_counts.append(job_count)

    # print out some useful information
    print '\n'
    print 'number of job descriptions:', total_job_count
    print 'number of job description sequences:', len(job_sequence_counts)
    print '\n'
    return data


def lda(job_descs, n_topics=200, n_words=15):
    """
    Runs the LDA algorithm, prints out the top 'n_words', and dumps the fitted LDA model to a
    pickled file.

    n_words: The number of words to print out from each discovered topic.
    n_topics: the number of topics to learn from the resume dataset.
    """

    # Import data
    # print '\n...Importing job description data.'
    # if 'resume_test_data.p' not in os.listdir('../test/'):
    #     build_dataset()
    #     sequence_data = p.load(open('../data/resume_data_train_test.p', 'rb'))
    # else:
    #     sequence_data = p.load(open('../data/resume_data_train_test.p', 'rb'))

    # parse data into LDA-usable format
    # data = flatten_descrip_seqs(job_desc_seqs_file)
    # job_sequence_counts = []
    # total_job_count = 0
    # for datum_idx, datum in enumerate(sequence_data):
    #     job_count = 0
    #     for job_descr_indx, job_description in enumerate(datum):
    #         data.append(' '.join(job_description))
    #         job_count += 1
    #         total_job_count += 1
    #     if job_count != 0:
    #         job_sequence_counts.append(job_count)
    #
    # # print out some useful information
    # print '\n'
    # print 'number of job descriptions:', total_job_count
    # print 'number of job description sequences:', len(job_sequence_counts)
    # print '\n'

    # Use tf (raw term count) features for LDA.
    print '...Extracting term frequency (bag of words) features for LDA.'
    tf_vectorizer = CountVectorizer()
    tf = tf_vectorizer.fit_transform(job_descs)

    p.dump(tf_vectorizer, open('../models/tf_vect.p', 'wb'))

    # save vocabulary for later use
    vocab = tf_vectorizer.vocabulary_

    # Build LDA model. Mimno paper uses 200 topics; otherwise, I'll keep the default scikit-learn
    # model parameters.  We can play with model parameters in order to investigate how they affect
    # results.
    print '...Building LDA model.'
    lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch',
                                          evaluate_every=10, n_jobs=16, verbose=10)

    start_time = timeit.default_timer()

    # Fit model to data
    print '...Fitting LDA model to job description text.\n'
    fitted_data = lda_model.fit_transform(tf)
    
    # save unique class labels for LDA topics
    class_components = lda_model.components_

    secs = timeit.default_timer() - start_time
    print "\nCompleted fitting LDA model to job description text in {} seconds.\n".format(secs)


    # view topics learned by the model
    print '...Viewing topics.\n'
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda_model, tf_feature_names, n_words)

    # save model to pickle file
    print '...Saving model.\n'
    p.dump(lda_model, open('../models/lda_' + str(n_topics) + '_topics.p', 'wb'))
    
    # # write fitted sequential data to pickle file
    # print '...saving fitted sequential data.\n'
    # fitted_sequential_data = [ [ fitted_data[idx] for idx in xrange(job_count) ]
    #                                                 for job_count in job_sequence_counts ]
    # print len(fitted_sequential_data)
    # print len([ job for datum in fitted_sequential_data for job in datum ])
    # p.dump([ [ fitted_data[idx] for idx in xrange(job_count) ]
    #                                 for job_count in job_sequence_counts ],
    #        open('../data/fitted_sequential_data' + str(n_topics) + '.p', 'wb'))
    
    for idx in xrange(5):
        # print out example output from LDA
        print job_descs[idx]
        print fitted_data[idx]
        print fitted_data.shape

        # plot the likelihood for each component for the example output
        plt.plot(fitted_data[idx])
        plt.title('Distribution over Topics')
        plt.xlabel('Topic Index')
        plt.ylabel('Likelihood')
        plt.show()

    print '\n'
    

if __name__ == '__main__':

    # n_words = raw_input('Enter number of words to print (default 15): ')
    # if n_words == '':
    #     n_words = 15
    # else:
    #     n_words = int(n_words)
    #
    # n_topics = raw_input('Enter number of topics to learn (default 100): ')
    # if n_topics == '':
    #     n_topics = 100
    # else:
    #     n_topics = int(n_topics)

    infile_name = sys.argv[1]
    n_topics = int(sys.argv[2])
    n_words = int(sys.argv[3])

    with open(infile_name, 'rb') as infile:
        job_desc_seqs = p.load(infile)

    job_descs = flatten_descrip_seqs(job_desc_seqs)

    lda(job_descs, n_topics, n_words)
