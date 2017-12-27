import sys
import math
import numpy as np
import resume_common as common


class ResumeHMM(object):
    def __init__(self, num_states, pi, gamma, num_topics):
        self.num_states = num_states
        self.pi = pi        # (uniform) prior on start state
        self.gamma = gamma  # (uniform) prior on state-state transitions

        self.sum_pi = pi*num_states
        self.sum_gamma = gamma*num_states

        self.start_counts = [0]*num_states
        self.state_trans = [[0]*num_states]*num_states
        self.state_trans_tots = [0]*num_states

        self.state_topic_counts = [[0]*num_states]*num_topics
        self.state_topic_totals = [0]*num_states

        self.sum_alpha = num_topics
        self.alphas = [self.sum_alpha/num_topics]*num_topics  # not sure why we do it this way... it's 1.0!

        self.num_sequences = None  # needed for denominator of likelihood, set in fit() below

    def fit(self, docs, num_states):
        self.num_sequences = len([ d for d in docs if d.doc_prev is None ])

    def sample_doc_state(self, doc):
        self.remove_from_trans_counts(doc)

        state_log_likes = [0.0]*self.num_states
        for s in range(len(self.num_states)):
            state_log_likes[s] = self.calc_state_state_log_like(doc, s) + \
                                 self.calc_state_topic_log_like(doc, s)

        # turn log likes into a distrib to sample from
        state_log_like_max = max(state_log_likes)
        state_samp_distrib = [ math.exp(lik - state_log_like_max) for lik in state_log_likes ]
        state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
        doc.state = state_new

        self.add_to_trans_counts(doc)
        self.add_to_topic_counts()

        return state_new

    def remove_from_trans_counts(self, doc):
        if doc.doc_prev is None:  # beginning of resume sequence
            self.start_counts[doc.state] -= 1
            if doc.doc_next is not None:  # not a singleton sequence
                self.state_trans[doc.state][doc.doc_next.state] -= 1
                self.state_trans_tots[doc.state] -= 1
        else:  # middle of sequence
            self.state_trans[doc.doc_prev.state][doc.state] -= 1

            if doc.doc_next is not None:  # not end of sequence
                self.state_trans[doc.state][doc.doc_next.state] -= 1
                self.state_trans_tots[doc.state] -= 1

    def calc_state_state_log_like(self, doc, s):
        if doc.doc_prev is None:  # beginning of resume sequence
            lik = (self.start_counts[s] + self.pi) / (self.num_sequences - 1 + self.sum_pi)

            if doc.doc_next is not None:  # not a singleton sequence
                lik *= self.state_trans[s][doc.doc_next.state] + self.gamma

        else:  # middle of sequence
            if doc.doc_next is None:  # end of sequence
                lik = (self.state_trans[doc.doc_prev.state][s] + self.gamma)
            else:
                if (doc.doc_prev.state == s) and (s == doc.doc_next.state):
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + 1 + self.gamma) /
                           (self.state_trans_tots[s] + 1 + self.sum_gamma))

                elif (doc.doc_prev.state == s): # and (s != doc_next.state):
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + self.gamma) /
                           (self.state_trans_tots[s] + 1 + self.sum_gamma))
                else: # (doc_prev.state != s)
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + self.gamma) /
                           (self.state_trans_tots[s] + self.sum_gamma))
        return math.log(lik)

    # todo: make this function cache itself
    def calc_state_topic_log_like(self, doc, state):
        ret = 0.0
        for topic, topic_count in enumerate(doc.topic_distrib):
            log_gammas = [0.0]
            for i in range(1, topic_count + 1):
                log_gammas.append(log_gammas[i-1] +
                                  math.log(self.alphas[topic] + i - 1 +
                                           self.state_topic_counts[state][topic]))
            ret += log_gammas[topic_count]

        log_gammas = [0.0]
        for i in range(1, doc.length + 1):
            log_gammas.append(log_gammas[i-1] +
                              math.log(self.sum_alpha + i - 1 + self.state_topic_totals[state]))
        ret -= log_gammas[doc.length]
        return ret

    def add_to_trans_counts(self, doc):
        if doc.doc_prev is None:  # beginning of resume sequence
            self.start_counts[doc.state] += 1

            if doc.doc_next is not None:  # not a singleton sequence
                self.state_trans[doc.state][doc.doc_next.state] += 1
                self.state_trans_tots[doc.state] += 1

        else:  # middle of sequence
            self.state_trans[doc.doc_prev.state][doc.state] += 1

            if doc.doc_next is not None:  # not the end of sequence
                self.state_trans[doc.state][doc.doc_next.state] += 1
                self.state_trans_tots[doc.state] += 1

    def add_to_topic_counts(self, doc):
        for topic, topic_count in enumerate(doc.topic_distrib):
            self.state_topic_counts[doc.state][topic] += topic_count
            self.state_topic_totals[doc.state] += doc.length




class Document(object):
    # start_state_counts = [0]*NUM_STATES
    # state_state_trans = []  #zzz todo: how the fuck?
    # state_trans_tots = []  #zzz is this necessary?

    # state_topic_counts = [][]  # when does this get initialized?
    # state_topic_totals = []

    def __init__(self, res_ent, prev=None, next=None):
        self.entry = res_ent
        self.doc_prev = prev
        self.doc_next = next

        #TODO: initialize these zzz
        self.topic_distrib = []  # one entry per topic
        self.length = None

        self.state = None

    # def sample_state(self):
    #     self.remove_from_trans_counts()
    #
    #     state_log_likes = [0.0]*NUM_STATES
    #     for s in range(NUM_STATES):
    #         state_log_likes[s] = self.calc_state_state_log_like(s) + \
    #                              self.calc_state_topic_log_like(s)
    #
    #     # turn log likes into a distrib to sample from
    #     state_log_like_max = max(state_log_likes)
    #     state_samp_distrib = [ math.exp(lik - state_log_like_max) for lik in state_log_likes ]
    #     state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
    #     self.state = state_new
    #
    #     self.add_to_trans_counts()
    #     self.add_to_topic_counts()

    # def add_to_topic_counts(self):
    #     for topic, topic_count in enumerate(self.topic_distrib):
    #         Document.state_topic_counts[self.state][topic] += topic_count
    #     Document.state_topic_totals[self.state] += self.length

    # def add_to_trans_counts(self):
    #     if self.doc_prev is None:  # beginning of resume sequence
    #         Document.start_state_counts[self.state] += 1
    #
    #         if self.doc_next is not None:  # not a singleton sequence
    #             Document.state_state_trans[self.state][self.doc_next.state] += 1
    #             Document.state_trans_tots[self.state] += 1
    #
    #     else:  # middle of sequence
    #         Document.state_state_trans[self.doc_prev.state][self.state] += 1
    #
    #         if self.doc_next is not None:  # not the end of sequence
    #             Document.state_state_trans[self.state][self.doc_next.state] += 1
    #             Document.state_trans_tots[self.state] += 1

    # def remove_from_trans_counts(self):
    #     if self.doc_prev is None:  # beginning of resume sequence
    #         Document.start_state_counts[self.state] -= 1
    #         if self.doc_next is not None:  # not a singleton sequence
    #             Document.state_state_trans[self.state][self.doc_next.state] -= 1
    #             Document.state_trans_tots[self.state] -= 1
    #     else:  # middle of sequence
    #         Document.state_state_trans[self.doc_prev.state][self.state] -= 1
    #
    #         if self.doc_next is not None:  # not end of sequence
    #             Document.state_state_trans[self.state][self.doc_next.state] -= 1
    #             Document.state_trans_tots[self.state] -= 1

    # def calc_state_state_log_like(self, s):
    #     if self.doc_prev is None:  # beginning of resume sequence
    #         lik = (Document.start_state_counts[s] + PI) / (numSequences - 1 + sumPI)
    #
    #         if self.doc_next is not None:  # not a singleton sequence
    #             lik *= Document.state_state_trans[s][self.doc_next.state] + gamma
    #
    #     else:  # middle of sequence
    #         if self.doc_next is None:  # end of sequence
    #             lik = (Document.state_state_trans[self.doc_prev.state][s] + gamma)
    #         else:
    #             if (self.doc_prev.state == s) and (s == self.doc_next.state):
    #                 lik = ((Document.state_state_trans[self.doc_prev.state][s] + gamma) *
    #                        (Document.state_state_trans[s][self.doc_next.state] + 1 + gamma) /
    #                        (Document.state_trans_tots[s] + 1 + gammaSum))
    #
    #             elif (self.doc_prev.state == s): # and (s != doc_next.state):
    #                 lik = ((Document.state_state_trans[self.doc_prev.state][s] + gamma) *
    #                        (Document.state_state_trans[s][self.doc_next.state] + gamma) /
    #                        (Document.state_trans_tots[s] + 1 + gammaSum))
    #             else: # (doc_prev.state != s)
    #                 lik = ((Document.state_state_trans[self.doc_prev.state][s] + gamma) *
    #                        (Document.state_state_trans[s][self.doc_next.state] + gamma) /
    #                        (Document.state_trans_tots[s] + gammaSum))
    #     return math.log(lik)


    # # todo: make this function cache itself
    # def calc_state_topic_log_like(self, state):
    #     ret = 0.0
    #     for topic, topic_count in enumerate(self.topic_distrib):
    #         log_gammas = [0.0]
    #         for i in range(1, topic_count + 1):
    #             log_gammas.append(log_gammas[i-1] +
    #                               math.log(alpha[topic] + i - 1 +
    #                                        Document.state_topic_counts[topic]))
    #         ret += log_gammas[topic_count]
    #
    #     log_gammas = [0.0]
    #     for i in range(1, self.length + 1):
    #         log_gammas.append(log_gammas[i-1] +
    #                           math.log(alpha_sum + i - 1 + Document.state_topic_totals[state]))
    #     ret -= log_gammas[self.length]
    #     return ret



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

# def load_docs_json(infile_name):
#     with open(infile_name, 'r') as infile:
#         docs_all = json.load(infile)
#     resumes = [ Resume(seq) for seq in docs_all ]
#     return resumes



def get_docs_from_resumes(resumes):
    docs = []
    for resume in resumes:
        if len(resume) == 1:
            docs.append(Document(resume[0], None, None))
        else:
            doc_0 = Document(resume[0])
            docs.append(doc_0)
            doc_prev = doc_0
            for res_ent in resume[1:]:
                doc = Document(res_ent, doc_prev)
                doc_prev.doc_next = doc
                docs.append(doc)
                doc_prev = doc
    return docs


def sample_states(docs, iterations):

    for iter in range(iterations):

        for doc in docs:

            doc.sample_state()


##########################################
if __name__ == '__main__':
    USAGE = "heya"

    resumes = common.load_json_file(sys.argv[1])  # gives us a list of lists of ResumeEntry
    resume_docs = get_docs_from_resumes(resumes)

    iters = int(sys.argv[2])
    sample_states(resume_docs, iters)










