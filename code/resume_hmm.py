import json
import math
import numpy as np

NUM_STATES = 42


class Resume(object):
    def __init__(self, docs):
        self.docs = docs


class Document(object):
    start_state_counts = [0]*NUM_STATES
    state_state_trans = []  #zzz todo: how the fuck?
    state_trans_tots = []  #zzz is this necessary?

    state_topic_counts = [][]  # when does this get initialized?
    state_topic_totals = []

    def __init__(self):
        self.state = None
        self.topic_distrib = []  # one entry per topic

    def sample_state(self, doc_prev, doc_next):

        self.remove_from_trans_counts(doc_prev, doc_next)

        state_log_likes = [0.0]*NUM_STATES
        for s in range(NUM_STATES):
            state_log_likes[s] = self.calc_state_state_log_like(s, doc_prev, doc_next) + \
                                 self.calc_state_topic_log_like(s)

        # turn log likes into a distrib to sample from
        state_log_like_max = max(state_log_likes)
        state_samp_distrib = [ math.exp(lik - state_log_like_max) for lik in state_log_likes ]
        state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
        self.state = state_new

        self.add_to_trans_counts(doc_prev, doc_next)
        self.add_to_topic_counts()

    # double sum = 0.0;
        # for (int state = 0; state < numStates; state++) {
        #     if (Double.isNaN(samplingDistribution[state])) {
        #         System.out.println(stateLogLikelihoods[state]);
        #     }
        #
        #     assert (!Double.isNaN(samplingDistribution[state]));
        #
        #     samplingDistribution[state] =
        #             Math.exp(stateLogLikelihoods[state] - max);
        #     sum += samplingDistribution[state];
        #
        #     if (Double.isNaN(samplingDistribution[state])) {
        #         System.out.println(stateLogLikelihoods[state]);
        #     }
        #
        #     assert (!Double.isNaN(samplingDistribution[state]));
        #
        #     if (doc % 100 == 0) {
        #         //System.out.println(samplingDistribution[state]);
        #     }
        # }
        #
        # int newState = r.nextDiscrete(samplingDistribution, sum);
        #
        # documentStates[doc] = newState;
        #
        # for (int topic = 0; topic < numTopics; topic++) {
        #     stateTopicCounts[newState][topic] += topicCounts.get(topic);
        # }
        # stateTopicTotals[newState] += docLength;
        # recacheStateTopicDistribution(newState, topicCounts);
        #

    def add_to_topic_counts(self):
        for topic, topic_count in enumerate(self.topic_distrib):
            Document.state_topic_counts[self.state][topic] += topic_count
        Document.state_topic_totals[self.state] += self.length

    def add_to_trans_counts(self, doc_prev, doc_next):
        if doc_prev is None:  # beginning of resume sequence
            Document.start_state_counts[self.state] += 1

            if doc_next is not None:  # not a singleton sequence
                Document.state_state_trans[self.state][doc_next.state] += 1
                Document.state_trans_tots[self.state] += 1

        else:  # middle of sequence
            Document.state_state_trans[doc_prev.state][self.state] += 1

            if doc_next is not None:  # not the end of sequence
                Document.state_state_trans[self.state][doc_next.state] += 1
                Document.state_trans_tots[self.state] += 1



    #
    # if (initializing) {
    #         // If we're initializing the states, don't bother
    #         //  looking at the next state.
    #
    #         if (previousSequenceID != sequenceID) {
    #             initialStateCounts[newState]++;
    #         } else {
    #             previousState = documentStates[doc - 1];
    #             stateStateTransitions[previousState][newState]++;
    #             stateTransitionTotals[newState]++;
    #         }
    #     } else {
    #         if (previousSequenceID != sequenceID && sequenceID != nextSequenceID) {
    #             // 1. This is a singleton document
    #
    #             initialStateCounts[newState]++;
    #         } else if (previousSequenceID != sequenceID) {
    #             // 2. This is the beginning of a sequence
    #
    #             initialStateCounts[newState]++;
    #
    #             nextState = documentStates[doc + 1];
    #             stateStateTransitions[newState][nextState]++;
    #             stateTransitionTotals[newState]++;
    #         } else if (sequenceID != nextSequenceID) {
    #             // 3. This is the end of a sequence
    #
    #             previousState = documentStates[doc - 1];
    #             stateStateTransitions[previousState][newState]++;
    #         } else {
    #             // 4. This is the middle of a sequence
    #
    #             previousState = documentStates[doc - 1];
    #             stateStateTransitions[previousState][newState]++;
    #
    #             nextState = documentStates[doc + 1];
    #             stateStateTransitions[newState][nextState]++;
    #             stateTransitionTotals[newState]++;
    #
    #         }
    #     }
    #





    def remove_from_trans_counts(self, doc_prev, doc_next):
        if doc_prev is None:  # beginning of resume sequence
            Document.start_state_counts[self.state] -= 1
            if doc_next is not None:  # not a singleton sequence
                Document.state_state_trans[self.state][doc_next.state] -= 1
                Document.state_trans_tots[self.state] -= 1
        else:  # middle of sequence
            Document.state_state_trans[doc_prev.state][self.state] -= 1

            if doc_next is not None:  # not end of sequence
                Document.state_state_trans[self.state][doc_next.state] -= 1
                Document.state_trans_tots[self.state] -= 1


    def calc_state_state_log_like(self, s, doc_prev, doc_next):

        if doc_prev is None:  # beginning of resume sequence
            lik = (Document.start_state_counts[s] + pi) / (numSequences - 1 + sumPI)

            if doc_next is not None:  # not a singleton sequence
                lik *= Document.state_state_trans[s][doc_next.state] + gamma

        else:  # middle of sequence
            if doc_next is None:  # end of sequence
                lik = (Document.state_state_trans[doc_prev.state][s] + gamma)
            else:
                if (doc_prev.state == s) and (s == doc_next.state):
                    lik = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                           (Document.state_state_trans[s][doc_next.state] + 1 + gamma) /
                           (Document.state_trans_tots[s] + 1 + gammaSum))

                elif (doc_prev.state == s): # and (s != doc_next.state):
                    lik = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                           (Document.state_state_trans[s][doc_next.state] + gamma) /
                           (Document.state_trans_tots[s] + 1 + gammaSum))
                else: # (doc_prev.state != s)
                    lik = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                           (Document.state_state_trans[s][doc_next.state] + gamma) /
                           (Document.state_trans_tots[s] + gammaSum))
        return math.log(lik)


    # todo: make this function cache itself
    def calc_state_topic_log_like(self, state):
        ret = 0.0
        for topic, topic_count in enumerate(self.topic_distrib):
            log_gammas = [0.0]
            for i in range(1, topic_count + 1):
                log_gammas.append(log_gammas[i-1] +
                                  math.log(alpha[topic] + i - 1 +
                                           Document.state_topic_counts[topic]))
            ret += log_gammas[topic_count]

        log_gammas = [0.0]
        for i in range(1, self.length + 1):
            log_gammas.append(log_gammas[i-1] +
                              math.log(alpha_sum + i - 1 + Document.state_topic_totals[state]))
        ret -= log_gammas[self.length]
        return ret



    #
    #
    #
    # def sample_initial_state(self, doc_prev, doc_next):
    #
    #     state_log_likes = []
    #
    #     for state in range(NUM_STATES):
    #         state_log_likes[state] =
    #
    #
    #
    #
    #     for (int state = 0; state < numStates; state++) {
    #     stateLogLikelihoods[state] = Math.log((initialStateCounts[state] + pi) /
    #     (numSequences - 1 + sumPi));
    #     }
    #

def load_docs_json(infile_name):
    with open(infile_name, 'r') as infile:
        docs_all = json.load(infile)
    resumes = [ Resume(seq) for seq in docs_all ]
    return resumes


def sample_states(iterations, docs):

    for iter in range(iterations):

        for doc in docs:

            doc.sample_state()

