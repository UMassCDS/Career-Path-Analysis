import cPickle as p
import numpy as np
from hmmlearn import base
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import lda
import os

def transform_input():
    '''
    Transforms the data extracted from the dataset into a single list that 
    is then used to "fit" a HMM. Pickles everything.

    ------
    Args: 
    none
    ------
    Returns:
    list_of_sequences: Likelihood sequences as a list 
    lengths_of_sequences: Length of each sequence
    '''
    if (not 'list_of_sequences.p' in os.listdir('../data')) or (not 'lengths_of_sequences.p' in os.listdir('../data')):
	print "Importing data..."
        hmm_input = p.load(open('../data/fitted_sequential_data_200.p'))
        print "Importing done!"
        list_of_sequences = []
	lengths_of_sequences = []
        print "Transforming data..."
        for sequence in hmm_input:
            for description in sequence:
		list_of_sequences.append(description.tolist())
	    lengths_of_sequences.append(len(sequence))
	p.dump(list_of_sequences,open('../data/list_of_sequences.p','wb'))
	p.dump(lengths_of_sequences,open('../data/lengths_of_sequences.p','wb'))
    else:
        print "Loading pickle dumps..."
        hmm_input = p.load(open('../data/fitted_sequential_data_200.p'))
        list_of_sequences = p.load(open('../data/list_of_sequences.p','rb'))
        lengths_of_sequences = p.load(open('../data/lengths_of_sequences.p','rb'))
	print "Pickle dumps loaded!"
    return list_of_sequences,lengths_of_sequences,hmm_input

def train_hmm(list_of_sequences,lengths_of_sequences,hmm_input,use_dump):
    
    '''
    Train a MultinomialHMM over job description sequences 
    -----
    Args:
    list_of_sequences: Transformed input to fit the HMM
    lengths_of_sequences: Lengths of sequences of likelihood lists
    hmm_input: List of sequences used in the prediction stage
    use_dump: Whether to use the existing HMM model, or fit a new one
    ------
    Returns:
    model: The final fitted MultinomialHMM model trained using EM
    encoder: The sklearn.preprocessing.LabelEncoder object used to transform sequences
    '''

    if  not 'hmm_100_components.p'  in os.listdir('../models') or use_dump == False:
        #print type(hmm_input),type(hmm_input[0]),type(hmm_input[0][0])
        #print type(list_of_sequences),type(list_of_sequences[0]),type(list_of_sequences[0][0])
        
        encoder = LabelEncoder()
        #Transform job descriptions 
        transformed_sequences = []
        for sequence in hmm_input:
            transformed_sequence=[]
            for description in sequence:
                transformed_sequence.append([np.argmax(description)])
            transformed_sequences.append(transformed_sequence)
        print transformed_sequences[0]
        #print type(transformed_sequences),type(transformed_sequences[0]),type(transformed_sequences[0][0])
        transformed_list = []
        for sequence in transformed_sequences:
            for element in sequence:
                for micro_element in element:
                    transformed_list.append(micro_element)
        print transformed_list[0]
        print type(transformed_list),type(transformed_list[0])
        transformed_transformed_list = encoder.fit_transform(transformed_list)
        end_point_train = sum(lengths_of_sequences[0:9034])
        train_data = transformed_transformed_list[0:end_point_train]
        test_data = transformed_transformed_list[end_point_train:]
        #print type(transformed_list),type(transformed_list[0]),type(transformed_list[0][0])
        
        i = 0
        transformed_descriptions_train = []
        while i<len(train_data):
            transformed_descriptions_train.append(train_data[i:i+1])
            i+=1
        
        i = 0
        transformed_descriptions_test = []
        while i<len(test_data):
            transformed_descriptions_test.append(test_data[i:i+1])
            i+=1
       
        test_lengths = lengths_of_sequences[9034:]
        transformed_test_data = []
        start = 0
        for length in test_lengths:
            end = start+length
            transformed_test_data.append(transformed_descriptions_test[start:end])
            start = end
        p.dump(transformed_test_data,open('../data/transformed_test_data.p','wb'))
    
        #Train a MultinomialHMM and pickle the model for future use
        model = hmm.MultinomialHMM(n_components = 100,n_iter = 15,verbose = True)
        print "Fitting HMM"
        model.fit(transformed_descriptions_train,lengths_of_sequences[0:9034])
        p.dump(model,open('../models/hmm_100_components.p','wb'))
        print "Fitting done!"
        
        model = p.load(open('../models/hmm_100_components.p','rb'))
        #print model.score(transformed_descriptions_train,lengths_of_sequences[0:9034])
        #print model.score(transformed_descriptions_test,lengths_of_sequences[9034:])
        return model,encoder,transformed_test_data
    else:
        model = p.load(open('../models/hmm_10_components.p','rb'))
        encoder = p.load(open('../models/encoder.p','rb'))
        transformed_test_data = p.load(open('../data/transformed_test_data.p','rb'))
        return model,encoder,transformed_test_data
    
def predict_next_transition(model,encoder,test_data,lengths_of_sequences):
    '''
    Function to predict what the topic of the next job description will be
    
    ------
    Args: 
    model: The fitted HMM used to predict next transition
    encoder: sklearn.preprocessing.LabelEncoder object used to encode the most likely topic
    test_data: The data to perform predcitions on
    lengths_of_sequences: The length of each sequence of job descriptions

    ------
    Returns:
    topics: Most likely topic for next job description
    '''


    #Determine sequences of hidden states that led to a particular sequence
    hidden_state_sequences = []
    print type(test_data),type(test_data[0]),type(test_data[0][0])
    print len(test_data),len(test_data[0]),len(test_data[0][0])
    
    bag_of_words = p.load(open('../data/resume_data_train_test.p'))
    bag_of_words_test = bag_of_words[9034:]
    for sequence in test_data:
        hidden_state_sequence = model.predict(sequence)
        hidden_state_sequences.append(hidden_state_sequence)

    #Find the possible subsequent hidden state
    next_states = []
    for sequence_of_states in hidden_state_sequences:
        last_state = sequence_of_states[-1]
        next_states.append(model.transmat_[last_state].argmax(axis = 0))
    
    #Generate emissions
    list_of_likelihoods = []
    for next_state in next_states:
        list_of_likelihoods.append(model._generate_sample_from_state(next_state))
    
    topics = []
    for emission in list_of_likelihoods:
        topics.append(encoder.inverse_transform(emission))
    return topics
    
    #print model.score(test_data,lengths_of_sequences[9034:])
    #print set(unique_data)
if __name__== '__main__':
        
    list_of_sequences,lengths_of_sequences,hmm_input = transform_input()
    model,encoder,test_data = train_hmm(list_of_sequences,lengths_of_sequences,hmm_input)
    next_topic = predict_next_transition(model,encoder,test_data,lengths_of_sequences)
