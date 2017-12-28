import math
import argparse
import numpy as np
from resume_lda import load_json_resumes_lda


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
        self.init_doc_states(docs)

    def init_doc_states(self, docs):
        for doc in docs:
            doc.state = self.sample_doc_state(doc)
            self.add_to_trans_counts(doc)
            self.add_to_topic_counts(doc)

    def sample_doc_states(self, docs, iterations):
        for i in range(iterations):
            for doc in docs:
                self.remove_from_trans_counts(doc)
                doc.state = self.sample_doc_state(doc)
                self.add_to_trans_counts(doc)
                self.add_to_topic_counts(doc)

    def sample_doc_state(self, doc):
        # self.remove_from_trans_counts(doc)

        state_log_likes = [0.0]*self.num_states
        for s in range(len(self.num_states)):
            state_log_likes[s] = self.calc_state_state_log_like(doc, s) + \
                                 self.calc_state_topic_log_like(doc, s)

        # turn log likes into a distrib to sample from
        state_log_like_max = max(state_log_likes)
        state_samp_distrib = [ math.exp(lik - state_log_like_max) for lik in state_log_likes ]
        state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)

        # doc.state = state_new
        # self.add_to_trans_counts(doc)
        # self.add_to_topic_counts()
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
    def __init__(self, res_ent, topic_distrib, prev=None, next=None):
        self.entry = res_ent
        self.topic_distrib = topic_distrib  # one entry per topic
        self.doc_prev = prev
        self.doc_next = next

        self.length = sum(topic_distrib)
        self.state = None


def get_docs_from_resumes(resumes):
    docs = []
    for resume in resumes:
        if len(resume) == 1:
            res_ent, top_dis = resume[0]
            docs.append(Document(res_ent, top_dis, prev=None, next=None))
        else:
            res_ent, top_dis = resume[0]
            doc_0 = Document(res_ent, top_dis)
            docs.append(doc_0)
            doc_prev = doc_0
            for res_ent, top_dis in resume[1:]:
                doc = Document(res_ent, top_dis, prev=doc_prev)
                doc_prev.doc_next = doc
                docs.append(doc)
                doc_prev = doc
    return docs


##########################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run multinomial HMM on resume data')
    parser.add_argument('infile', metavar='res_lda.json')
    parser.add_argument('num_states', type=int)
    parser.add_argument('num_iters', type=int)
    parser.add_argument('--pi', type=int, default=1000)
    parser.add_argument('--gamma', type=int, default=1)

    args = parser.parse_args()

    # get a list of lists of (ResumeEntry, topic_distrib) tuples
    resumes = load_json_resumes_lda(args.infile)
    resume_docs = get_docs_from_resumes(resumes)

    hmm = ResumeHMM(args.num_states, args.pi, args.gamma, args.num_topics)
    hmm.init_doc_states(resume_docs)
    hmm.sample_doc_states(resume_docs, args.num_iters)









