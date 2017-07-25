"""
Template for running LDA on job description corpus.

@author: Dan Saunders (djsaunde.github.io)
"""

# todo: add logging support

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


def get_top_words(words, prob_distrib, num):
    word_indices_sorted = prob_distrib.argsort()[::-1]

    rets = []
    for word_idx in word_indices_sorted[:num]:
        word = words[word_idx]
        word_prob = prob_distrib[word_idx]
        rets.append((word, word_prob))
    return rets


def print_top_words(components_norm, feature_names, n_words, n_topic=0):
    """
    Prints top 'n_words' words from the topics 'feature_names'.

    model: The LDA model from which to print representative words from topics.
    feature_names: The names of the representative word tokens.
    n_words: The number of representative words to print for each topic.
    """
    # if n_topic == 0:
    #     for topic_idx, topic in enumerate(model.components_):
    #         print 'Topic #%d:' % topic_idx
    #         print ' '.join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
    #     print '\n'
    # else:
    #     for topic_idx, topic in enumerate(model.components_):
    #         if topic_idx == n_topic:
    #             print 'Topic #%d:' % topic_idx
    #             print ' '.join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])

    for topic_idx, topic in enumerate(components_norm):

        print 'topic #%d:' % topic_idx

        word_indices_sorted = topic.argsort()[::-1]
        for word_idx in word_indices_sorted[:n_words]:
            word = feature_names[word_idx]
            word_freq = topic[word_idx]

            try:
                print "\t", word, ":\t", word_freq
            except UnicodeEncodeError:
                print "\t", word.encode('ascii', 'ignore'), ":\t", word_freq


def flatten_descrip_seqs(timelines):
    data = []
    job_sequence_counts = []
    total_job_count = 0
    for t, jobs in enumerate(timelines):
        job_count = 0
        for j, (start, end, company_name, desc) in enumerate(jobs):
            data.append(' '.join(desc))
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


def lda(job_descs, n_topics=200, n_jobs=16, n_words=15):
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

    # p.dump(tf_vectorizer, open('../models/tf_vect.p', 'wb'))

    # save vocabulary for later use
    vocab = tf_vectorizer.vocabulary_

    # Build LDA model. Mimno paper uses 200 topics; otherwise, I'll keep the default scikit-learn
    # model parameters.  We can play with model parameters in order to investigate how they affect
    # results.
    print '...Building LDA model.'
    lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch',
                                          evaluate_every=10, n_jobs=n_jobs, verbose=10)

    start_time = timeit.default_timer()

    # Fit model to data
    print '...Fitting LDA model to job description text.\n'
    fitted_data = lda_model.fit_transform(tf)
    
    # # save unique class labels for LDA topics
    # class_components = lda_model.components_


    # need to normalize to get probabilities, see:
    # https: // github.com / scikit - learn / scikit - learn / issues / 6353
    # https://github.com/scikit-learn/scikit-learn/pull/8805/commits/ceab61ce78cddfa8f2975989730a8fbc3fc76ada
    components_norm = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]

    secs = timeit.default_timer() - start_time
    print "\nCompleted fitting LDA model to job description text in {} seconds.\n".format(secs)


    # view topics learned by the model
    print '...Viewing topics.\n'
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(components_norm, tf_feature_names, n_words)

    print "\n\n"

    # # save model to pickle file
    # print '...Saving model.\n'
    # p.dump(lda_model, open('../models/lda_' + str(n_topics) + '_topics.p', 'wb'))
    
    # # write fitted sequential data to pickle file
    # print '...saving fitted sequential data.\n'
    # fitted_sequential_data = [ [ fitted_data[idx] for idx in xrange(job_count) ]
    #                                                 for job_count in job_sequence_counts ]
    # print len(fitted_sequential_data)
    # print len([ job for datum in fitted_sequential_data for job in datum ])
    # p.dump([ [ fitted_data[idx] for idx in xrange(job_count) ]
    #                                 for job_count in job_sequence_counts ],
    #        open('../data/fitted_sequential_data' + str(n_topics) + '.p', 'wb'))
    
    for desc_idx in range(10, 30):
        # print out example output from LDA
        print "job desc {}: {}".format(desc_idx, job_descs[desc_idx])

        print_threshold = 0.01
        topic_indices_sorted = fitted_data[desc_idx].argsort()[::-1]
        for t, topic_idx in enumerate(topic_indices_sorted):

            if fitted_data[desc_idx][topic_idx] < print_threshold:
                break

            print "\t{}. topic {} ({}):".format(t, topic_idx, fitted_data[desc_idx][topic_idx])
            word_prob_tups = get_top_words(tf_feature_names, components_norm[topic_idx], 10)
            for word, prob in word_prob_tups:
                print "\t\t", word, ":\t", prob
            print "\n"

        # print fitted_data[idx]
        # print fitted_data.shape

        #todo: print top x words for all topics that are above some threshold for the job desc doc



        # # plot the likelihood for each component for the example output
        # plt.plot(fitted_data[idx])
        # plt.title('Distribution over Topics')
        # plt.xlabel('Topic Index')
        # plt.ylabel('Likelihood')
        # plt.show()

    print '\n'
    

if __name__ == '__main__':

    NUM_WORDS_PRINT = 20

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

    USAGE = " usage: " + sys.argv[0] + " job_desc_seqs.p num_topics num_jobs"
    if len(sys.argv) < 4:
        sys.exit(USAGE)

    infile_name = sys.argv[1]
    num_topics = int(sys.argv[2])
    num_jobs = int(sys.argv[3])

    with open(infile_name, 'rb') as infile:
        job_desc_seqs = p.load(infile)

    job_descs = flatten_descrip_seqs(job_desc_seqs)

    lda(job_descs, num_topics, num_jobs, NUM_WORDS_PRINT)




