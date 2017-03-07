'''
Template for running LDA on job description corpus.

@author: Dan Saunders (djsaunde.github.io)
'''

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from parse import xmlToDataset

import os, timeit
import cPickle as p
import pandas as pd


def print_top_words(model, feature_names, n_words):
    '''
    Prints top 'n_words' words from the topics 'feature_names'.
    '''
    for topic_idx, topic in enumerate(model.components_):
        print 'Topic #%d:' % topic_idx
        print ' '.join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
    print '\n'

# Number of words to view from each topic
n_words = 15
# Number of topics to construct (Mimno paper used 200)
n_topics = 200

# Import data
print '\n...Importing job description data.'
if not 'resume_data.p' in os.listdir('../data/'):
	xmlToDataset()
	data = p.load(open('../data/resume_data.p', 'rb'))
else:
	data = p.load(open('../data/resume_data.p', 'rb'))

# Use tf (raw term count) features for LDA.
print '...Extracting term frequency (bag of words) features for LDA.'
tf_vectorizer = CountVectorizer()
tf = tf_vectorizer.fit_transform(data)

# Build LDA model. Mimno paper uses 200 topics; otherwise, I'll keep the default scikit-learn model parameters.
# We can play with model parameters in order to investigate how they affect results
print '...Building LDA model.'
lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch', evaluate_every=10, n_jobs=-1, verbose=10)

start_time = timeit.default_timer()

# Fit model to data
print '...Fitting LDA model to job description text.\n'
lda_model.fit(tf)

print '\nCompleted fitting LDA model to job description text in ' + str(timeit.default_timer() - start_time) + ' seconds.\n'

# view topics learned by the model
print '...Viewing topics.\n'
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda_model, tf_feature_names, n_words)

# save model to pickle file
print '...Saving model.\n'
p.dump(lda_model, open('../models/lda_' + str(n_topics) + '_topics.p', 'wb'))
