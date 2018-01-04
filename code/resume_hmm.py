import sys
import math
import argparse
import json
import datetime
import os.path
import numpy as np
from resume_lda import load_json_resumes_lda


OUT_PARAMS = 'params.tsv'
OUT_STATE_TRANS = 'trans.tsv'
OUT_STATE_TOPICS = 'topics.tsv'
OUT_STATES = 'states.tsv'


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

        self.state_topic_counts = [[0]*num_topics]*num_states
        self.state_topic_totals = [0]*num_states

        self.sum_alpha = num_topics
        self.alphas = [self.sum_alpha/num_topics]*num_topics  #zzz Why do it this way? It's 1.0!

        self.num_sequences = None  # needed for denominator of likelihood, set in fit() below

    # def fit(self, docs, iters, save_dir=None):
    #     self.num_sequences = len([ d for d in docs if d.doc_prev is None ])
    #     self.init_doc_states(docs)
    #     self.sample_doc_states(docs, save_dir, iters, iters_lag, )

    def fit(self, docs, save_dir, iters, iters_lag):
        # noinspection PyPep8,PyPep8
        self.num_sequences = len([ d for d in docs if d.doc_prev is None ])

        if os.path.isfile(os.path.join(save_dir, OUT_PARAMS)):
            i = self.load_progress(docs, save_dir)
            self.sample_doc_states(docs, save_dir, iters, iters_lag, start_iter=i+1)
        else:
            self.init_doc_states(docs)
            self.sample_doc_states(docs, save_dir, iters, iters_lag)

    def init_doc_states(self, docs):
        for doc in docs:
            # doc.state = self.sample_doc_state(doc)

            state_log_likes = [0.0] * self.num_states
            for s in range(self.num_states):
                state_log_likes[s] = self.init_state_log_like(doc, s) + \
                                     self.calc_state_topic_log_like(doc, s)
            doc.state = sample_from_loglikes(state_log_likes)

            self.init_trans_counts(doc)
            self.add_to_topic_counts(doc)

    def save_progress(self, i, docs, save_dir):
        ts = str(datetime.datetime.now())

        # model parameters (should not change between iters)
        params = {
            "num_states": self.num_states,
            "pi": self.pi,
            "gamma": self.gamma,
            "alphas": self.alphas,
            "num_sequences": self.num_sequences,
        }
        json_str = json.dumps(params)
        append_to_file(os.path.join(save_dir, OUT_PARAMS), [ts, i, json_str])

        # data structures capturing results of latest sampling iteration
        json_str = json.dumps(self.state_trans)
        append_to_file(os.path.join(save_dir, OUT_STATE_TRANS), [ts, i, json_str])

        json_str = json.dumps(self.state_topic_counts)
        append_to_file(os.path.join(save_dir, OUT_STATE_TOPICS), [ts, i, json_str])

        json_str = json.dumps([doc.state for doc in docs])
        append_to_file(os.path.join(save_dir, OUT_STATES), [ts, i, json_str])

    def load_progress(self, docs, save_dir):
        ts, iter_params, json_str = read_last_line(os.path.join(save_dir, OUT_PARAMS))
        params = json.loads(json_str)
        self.num_states = params["num_states"]
        self.pi = params["pi"]
        self.gamma = params["gamma"]
        self.alphas = params["alphas"]
        self.num_sequences = params["num_sequences"]

        ts, iter_trans, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TRANS))
        self.state_trans = json.loads(json_str)

        ts, iter_topics, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TOPICS))
        self.state_topic_counts = json.loads(json_str)

        ts, iter_states, json_str = read_last_line(os.path.join(save_dir, OUT_STATES))
        doc_states = json.loads(json_str)
        for doc, state in zip(docs, doc_states):
            doc.state = state

        if iter_params == iter_trans == iter_topics == iter_states:
            return iter_params
        else:
            sys.exit("unequal iter counts loaded")

    def sample_doc_states(self, docs, save_dir, iterations, lag_iters, start_iter=0):
        for i in range(start_iter, iterations):
            for d, doc in enumerate(docs):
                print "iter ", i, ", doc ", d
                self.remove_from_trans_counts(doc)

                # doc.state = self.sample_doc_state(doc)
                state_log_likes = [0.0] * self.num_states
                for s in range(self.num_states):
                    state_log_likes[s] = self.calc_state_state_log_like(doc, s) + \
                                         self.calc_state_topic_log_like(doc, s)
                doc.state = sample_from_loglikes(state_log_likes)

                self.add_to_trans_counts(doc)
                self.add_to_topic_counts(doc)
                print ""

            if i % lag_iters == 0:
                self.save_progress(i, docs, save_dir)
                # self.dump_sampled_state(str(i), docs)

    # def dump_sampled_state(self, tag, docs):
    #     json_str = json.dumps(self.state_trans)
    #     with open(self.outfile_state_trans, 'a') as out:
    #         out.write(tag + "\t" + json_str + "\n")
    #
    #     json_str = json.dumps(self.state_topic_counts)
    #     with open(self.outfile_state_topic_counts, 'a') as out:
    #         out.write(tag + "\t" + json_str + "\n")
    #
    #     json_str = json.dumps([doc.state for doc in docs])
    #     with open(self.outfile_states, 'a') as out:
    #         out.write(tag + "\t" + json_str + "\n")

    # def sample_doc_state(self, doc):
    #     state_log_likes = [0.0]*self.num_states
    #     for s in range(self.num_states):
    #         state_log_likes[s] = self.calc_state_state_log_like(doc, s) + \
    #                              self.calc_state_topic_log_like(doc, s)
    #     state_new = sample_from_loglikes(state_log_likes)
    #     return state_new

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

    def init_state_log_like(self, doc, s):
        # this  is just like calc_state_state_log_like(), except we don't have access to
        # the state of doc_next while initializing, so things are a bit simpler
        if doc.doc_prev is None:  # beginning of resume sequence
            lik = (self.start_counts[s] + self.pi) / (self.num_sequences - 1 + self.sum_pi)
        else:
            lik = (self.state_trans[doc.doc_prev.state][s] + self.gamma)
        return math.log(lik)

    def init_trans_counts(self, doc):
        if doc.doc_prev is None:  # beginning of resume sequence
            self.start_counts[doc.state] += 1
        else:  # middle of sequence
            self.state_trans[doc.doc_prev.state][doc.state] += 1
            if doc.doc_next is not None:  # not the end of sequence
                self.state_trans_tots[doc.state] += 1

    def calc_state_state_log_like(self, doc, s):
        trace = ""
        if doc.doc_prev is None:  # beginning of resume sequence
            trace += "begin "
            lik = (self.start_counts[s] + self.pi) / (self.num_sequences - 1 + self.sum_pi)
            trace += "({} + {}) / ({} - 1 + {}) = {}".format(self.start_counts[s], self.pi,
                                                             self.num_sequences, self.sum_pi,
                                                             lik)
            if doc.doc_next is not None:  # not a singleton sequence
                trace += "cont "
                print "state ", s, " doc_next.state ", doc.doc_next.state
                lik *= self.state_trans[s][doc.doc_next.state] + self.gamma

        else:  # middle of sequence
            trace += "middle "
            if doc.doc_next is None:  # end of sequence
                lik = (self.state_trans[doc.doc_prev.state][s] + self.gamma)
                trace += "end "
            else:
                trace += "cont "
                if (doc.doc_prev.state == s) and (s == doc.doc_next.state):
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + 1 + self.gamma) /
                           (self.state_trans_tots[s] + 1 + self.sum_gamma))

                elif doc.doc_prev.state == s:  # and (s != doc_next.state):
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + self.gamma) /
                           (self.state_trans_tots[s] + 1 + self.sum_gamma))
                else:  # (doc_prev.state != s)
                    lik = ((self.state_trans[doc.doc_prev.state][s] + self.gamma) *
                           (self.state_trans[s][doc.doc_next.state] + self.gamma) /
                           (self.state_trans_tots[s] + self.sum_gamma))
        print "lik for state ", s, "(", trace, ")", ": ", lik
        return math.log(lik)

    # todo: make this function cache itself
    def calc_state_topic_log_like(self, doc, state):
        ret = 0.0
        for topic, topic_count_float in enumerate(doc.topic_distrib):
            topic_count = int(round(topic_count_float))
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
    def __init__(self, res_ent, topic_distrib, doc_prev=None, doc_next=None):
        self.entry = res_ent
        self.topic_distrib = topic_distrib  # one entry per topic
        self.doc_prev = doc_prev
        self.doc_next = doc_next

        self.length = int(round(sum(topic_distrib)))
        self.state = None


def sample_from_loglikes(state_log_likes):
    # turn log likes into a distrib to sample from
    state_log_like_max = max(state_log_likes)
    # noinspection PyPep8,PyPep8
    state_likes_divmax = [ math.exp(loglik - state_log_like_max) for loglik in state_log_likes ]
    norm = sum(state_likes_divmax)
    # noinspection PyPep8,PyPep8
    state_samp_distrib = [ lik/norm for lik in state_likes_divmax ]
    # print state_log_likes
    # print state_samp_distrib
    state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
    return state_new


def get_docs_from_resumes(resume_list):
    docs = []
    for resume in resume_list:
        if len(resume) == 1:
            res_ent, top_dis = resume[0]
            docs.append(Document(res_ent, top_dis, doc_prev=None, doc_next=None))
        else:
            res_ent, top_dis = resume[0]
            doc_0 = Document(res_ent, top_dis)
            docs.append(doc_0)
            doc_prev = doc_0
            for res_ent, top_dis in resume[1:]:
                doc = Document(res_ent, top_dis, doc_prev=doc_prev)
                doc_prev.doc_next = doc
                docs.append(doc)
                doc_prev = doc
    return docs


def append_to_file(file_name, elts):
    with open(file_name, 'a') as out:
        out.write("\t".join(elts) + "\n")


def read_last_line(file_name):
    line = None
    with open(file_name, 'r') as infile:
        for line in infile:
            pass
        return line.rstrip("\n").split("\t")


##########################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run multinomial HMM on resume data')
    parser.add_argument('infile', metavar='res_lda.json')
    parser.add_argument('savedir', metavar='/progress/save/dir')
    parser.add_argument('num_states', type=int)
    parser.add_argument('num_iters', type=int)
    parser.add_argument('--pi', type=float, default=1000.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lag', type=int, default=10)

    args = parser.parse_args()

    # get a list of lists of (ResumeEntry, topic_distrib) tuples
    resumes = load_json_resumes_lda(args.infile)
    num_tops = len(resumes[0][0][1])  # distrib for the first job entry in the first resume
    resume_docs = get_docs_from_resumes(resumes)

    hmm = ResumeHMM(args.num_states, args.pi, args.gamma, num_tops)
    # hmm.init_doc_states(resume_docs)
    # hmm.sample_doc_states(resume_docs, args.num_iters)
    hmm.fit(resume_docs, args.savedir, args.num_iters, args.lag)
