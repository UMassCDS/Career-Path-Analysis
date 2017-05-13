import hmm
import lda

if __name__ == '__main__':

    '''
    #LDA part
    n_words = raw_input('Enter number of words to print (default 15): ')
    if n_words == '':
        n_words = 15
    else:
        n_words = int(n_words)

    n_topics = raw_input('Enter number of topics to learn (default 100): ')
    if n_topics == '':
        n_topics = 100
    else:
        n_topics = int(n_topics)
    lda.lda(n_words,n_topics)
    '''
    #HMM part
    list_of_sequences,lengths_of_sequences, hmm_input = hmm.transform_input()
    
    use_dump_flag = raw_input('Enter 1 to use dump, 0 to train new HMM')

    if int(use_dump_flag) == 1:
        model,encoder,test_data = hmm.train_hmm(list_of_sequences,lengths_of_sequences,hmm_input,True)
    elif int(use_dump_flag) == 0:
        model,encoder,test_data = hmm.train_hmm(list_of_sequences,lengths_of_sequences,hmm_input,False)

    next_topics = hmm.predict_next_transition(model,encoder,test_data, lengths_of_sequences)
