'''
Template for running LDA on job description corpus.

@author: Dan Saunders (djsaunde.github.io)
'''

from sklearn.decomposition import LatentDirichletAllocation

def print_top_words(model, feature_names, n):
    '''
    Prints top 'n' words from the topics 'feature_names'.
    '''
    for topic_idx, topic in enumerate(model.components_):
        print 'Topic #%d:' % topic_idx
        print ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print '\n'

# number of words to view from each topic
n_words = 15
# number of topics to construct (Mimno paper used 200)
n_topics = 200

# Import data (TODO)
print '...Importing job description data.'
data = # call Suraj's data import module here

# Use tf (raw term count) features for LDA.
print '...Extracting term frequency (bag of words) features for LDA.'
tf_vectorizer = CountVectorizer()
tf = tf_vectorizer.fit_transform(data)

# Build LDA model. Mimno paper uses 200 topics; otherwise, I'll keep the default scikit-learn model parameters.
# We can play with model parameters in order to investigate how they affect results
print '...Building LDA model.'
lda_model = LatentDirichletAllocation(n_topics=n_topics)

# Fit model to data (TODO); data should have shape (n_samples, n_features) and should be a "document word matrix".
# I believe this is a bag of words per document, in a word vector (one-hot) format
print '...Fitting LDA model to job description data.'
lda_model.fit(data)

# view topics learned by the model
print '...Viewing topics.\n'
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_words)
