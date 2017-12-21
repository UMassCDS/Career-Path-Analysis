import json
import math

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
        state_likes = []  # the likelihood for each possible new state

        if doc_prev is None:  # beginning of resume sequence

            Document.start_state_counts[self.state] -= 1

            for s in range(NUM_STATES):
                state_likes[s] = (Document.start_state_counts[s] + pi) / (numSequences - 1 + sumPI)

            if doc_next is not None:  # not a singleton sequence
                Document.state_state_trans[self.state][doc_next.state] -= 1
                Document.state_trans_tots[self.state] -= 1

                for s in range(NUM_STATES):
                    state_likes[s] *= Document.state_state_trans[s][doc_next.state] + gamma

        else:  # middle of sequence

            if doc_next is None:  # end of sequence

                Document.state_state_trans[doc_prev.state][self.state] -= 1

                for s in range(NUM_STATES):
                    state_likes[s] = (Document.state_state_trans[doc_prev.state][s] + gamma)

            else:

                Document.state_state_trans[doc_prev.state][self.state] -= 1
                Document.state_state_trans[self.state][doc_next.state] -= 1
                Document.state_trans_tots[self.state] -= 1

                for s in range(NUM_STATES):
                    if (doc_prev.state == s) and (s == doc_next.state):
                        state_likes[s] = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                                          (Document.state_state_trans[s][doc_next.state] + 1 + gamma) /
                                          (Document.state_trans_tots[s] + 1 + gammaSum))

                    elif (doc_prev.state == s): # and (s != doc_next.state):
                        state_likes[s] = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                                          (Document.state_state_trans[s][doc_next.state] + gamma) /
                                          (Document.state_trans_tots[s] + 1 + gammaSum))
                    else: # (doc_prev.state != s)
                        state_likes[s] = ((Document.state_state_trans[doc_prev.state][s] + gamma) *
                                          (Document.state_state_trans[s][doc_next.state] + gamma) /
                                          (Document.state_trans_tots[s] + gammaSum))

        state_log_likes = [ math.log(x) for x in state_likes ]

        for s in range(NUM_STATES):
            state_log_likes[s] += self.calc_state_topic_log_like(s)

            state_log_like_max = max(state_log_like_max, state_log_likes[s])


        # turn log likes into a distrib to sample from
        state_log_likes_sum = 0.0
        # for s in range(NUM_STATES):
        #
        #
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






    def sample_initial_state(self, doc_prev, doc_next):

        state_log_likes = []

        for state in range(NUM_STATES):
            state_log_likes[state] =




        for (int state = 0; state < numStates; state++) {
        stateLogLikelihoods[state] = Math.log((initialStateCounts[state] + pi) /
        (numSequences - 1 + sumPi));
        }


def load_docs_json(infile_name):
    with open(infile_name, 'r') as infile:
        docs_all = json.load(infile)
    resumes = [ Resume(seq) for seq in docs_all ]
    return resumes


def sample_states(iterations, docs):

    for iter in range(iterations):

        for doc in docs:

            doc.sample_state()