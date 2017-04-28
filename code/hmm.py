import cPickle as p
import numpy as np
from hmmlearn import base
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import os
'''
TODO: Get predictions up and running 
'''
def transform_input():
    '''
    Transforms the data extracted from the dataset into a single list that 
    is then used to "fit" a HMM. Pickles everything.

    ------
    Args: 
    none
    ------
    Returns:
    listOfSequences: Likelihood sequences as a list 
    lengthsOfSequences: Length of each sequence
    '''
    if not 'listOfSequences.p' in os.listdir('../data'):
	print "Importing data..."
        hmmInput = p.load(open('../data/fitted_sequential_data_200.p'))
        print "Importing done!"
        listOfSequences = []
	lengthsOfSequences = []
        print "Transforming data..."
        for sequence in hmmInput:
            for description in sequence:
		listOfSequences.append(description.tolist())
	    lengthsOfSequences.append(len(sequence))
	p.dump(listOfSequences,open('../data/listOfSequences.p','wb'))
	p.dump(lengthsOfSequences,open('../data/lengthsOfSequences.p','wb'))
    else:
        print "Loading pickle dumps..."
        hmmInput = p.load(open('../data/fitted_sequential_data_200.p'))
        listOfSequences = p.load(open('../data/listOfSequences.p','rb'))
        lengthsOfSequences = p.load(open('../data/lengthsOfSequences.p','rb'))
	print "Pickle dumps loaded!"
    return listOfSequences,lengthsOfSequences,hmmInput

def train_hmm(listOfSequences,lengthsOfSequences,hmmInput):
    '''
    Train a MultinomialHMM over job description sequences 
    -----
    Args:
    listOfSequences: Transformed input to fit the HMM
    lengthsOfSequences: Lengths of sequences of likelihood lists
    hmmInput: List of sequences used in the prediction stage
    
    ------
    Returns:
    model: The final fitted MultinomialHMM model trained using EM
    encoder: The sklearn.preprocessing.LabelEncoder object used to transform sequences
    '''
    if  not 'hmm_100_components.p'  in os.listdir('../models'):
        #print type(hmmInput),type(hmmInput[0]),type(hmmInput[0][0])
        #print type(listOfSequences),type(listOfSequences[0]),type(listOfSequences[0][0])
        
        encoder = LabelEncoder()
        #Transform job descriptions 
        transformed_sequences = []
        for sequence in hmmInput:
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
        #print type(transformed_list),type(transformed_list[0]),type(transformed_list[0][0])
        
        i = 0
        transformed_transformed_descriptions = []
        while i<len(transformed_transformed_list):
            transformed_transformed_descriptions.append(transformed_transformed_list[i:i+1])
            i+=1
        
        #Train a MultinomialHMM and pickle the model for future use
        model = hmm.MultinomialHMM(n_components = 100,n_iter = 1000,verbose = True)
        print "Fitting HMM"
        model.fit(transformed_transformed_descriptions,lengthsOfSequences)
        p.dump(model,open('../models/hmm_10_components.p','wb'))
        print "Fitting done!"
        
    else:
        model = p.load(open('../models/hmm_10_components.p','rb'))
        encoder = p.load(open('../models/encoder.p','rb'))
    return model,encoder
'''    
def predict_next_transition(model,encoder,hmmInput):
    
    #Determine sequences of states that led to a particular sequence
    hidden_state_sequences = []
    print type(hmmInput),type(hmmInput[0]),type(hmmInput[0][0])
    print len(hmmInput),len(hmmInput[0]),len(hmmInput[0][0])
    for sequence in hmmInput[0:2]:
        transformed_sequence = []
        for element in sequence:
            print element
            print len(element)
            transformed_element = encoder.transform(element)
            print len(transformed_element)
            transformed_sequence.append(transformed_element.tolist())
        print transformed_sequence
        print type(transformed_sequence),type(transformed_sequence[0])
        print len(transformed_sequence),len(transformed_sequence[0])
        hidden_state_sequence = model.predict(transformed_sequence)
        #hidden_state_sequences.append(hidden_state_sequence)
    #print hiddenStateSequences

    #Find the possible subsequent hidden state
    nextStates = []
    for sequence_of_states in hidden_state_sequences:
        last_state = sequence_of_states[-1]
        next_states.append(model.transmat_[last_state].argmax(axis = 0))
    print next_states
    
    #Generate emissions
    list_of_likelihoods = []
    for next_state in next_states:
        list_of_likelihoods.append(model._generate_sample_from_state(next_state))
    print np.sum(hmmInput[0][0])
    print np.sum(encoder.inverse_transform(list_of_likelihoods[0]))
'''

listOfSequences,lengthsOfSequences,hmmInput = transform_input()
train_hmm(listOfSequences,lengthsOfSequences,hmmInput)
#predict_next_transition(model,encoder,hmmInput)
