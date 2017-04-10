import cPickle as p
import numpy as np
from hmmlearn import base
from hmmlearn import hmm
import os

def transformInput():
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
    if not 'listsOfSequences.p' in os.listdir('../data'):
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
        listOfSequences = p.load(open('../data/listOfSequences.p','rb'))
        lengthsOfSequences = p.load(open('../data/lengthsOfSequences.p','rb'))
	
    return listOfSequences,lengthsOfSequences,hmmInput

def predictNextTransition(listOfSequences,lengthsOfSequences,hmmInput):
    '''
    Generates the next likelihood list(not sure if the answer is right, though)

    -----
    Args:
    listOfSequences: Transformed input to fit the HMM
    lengthsOfSequences: Lengths of sequences of likelihood lists
    hmmInput: List of sequences used in the prediction stage

    ------
    Returns:
    Not yet finalized
    '''
    #Learn a HMM from the sequences
    model = hmm.GaussianHMM(n_components = 10,n_iter = 100)
    model.fit(listOfSequences,lengthsOfSequences)
    
    #Determine sequences of hidden states that led to a particular sequence
    hiddenStateSequences = []

    for sequence in hmmInput:
        hiddenStateSequence = model.predict(sequence)
        hiddenStateSequences.append(hiddenStateSequence)
    print hiddenStateSequences

    #Find the possible subsequent hidden state
    nextStates = []
    for sequenceOfStates in hiddenStateSequences:
        lastState = sequenceOfStates[-1]
        nextStates.append(model.transmat_[lastState].argmax(axis = 0))
    print nextStates
    
    #Generate the next likelihood list
    listOfLikelihoods = []
    for nextState in nextStates:
        listOfLikelihoods.append(model._generate_sample_from_state(nextState))
    print np.sum(hmmInput[0][0])
    print np.min(listOfLikelihoods[0])


listOfSequences,lengthsOfSequences,hmmInput = transformInput()

predictNextTransition(listOfSequences,lengthsOfSequences,hmmInput)
