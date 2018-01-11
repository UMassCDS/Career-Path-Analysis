import sys
import math
import argparse
import json
import datetime
import logging
import os
import os.path
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Pool
from resume_lda import load_json_resumes_lda

import scipy.special

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


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

        self.state_topic_counts = [np.zeros(num_topics)]*num_states
        self.state_topic_totals = [0]*num_states

        self.sum_alpha = num_topics
        self.alphas = np.array([self.sum_alpha/num_topics]*num_topics)  #zzz Why do it this way? It's 1.0!

        self.num_sequences = None  # needed for denominator of likelihood, set in fit() below

    def fit(self, docs, save_dir, iters, iters_lag, erase=False):
        self.num_sequences = len([ d for d in docs if d.doc_prev is None ])

        if erase:
            self.delete_progress(save_dir)
        if os.path.isfile(os.path.join(save_dir, OUT_PARAMS)):
            i = self.load_progress(docs, save_dir)
            self.sample_doc_states(docs, save_dir, iters, iters_lag, start_iter=i+1)
        else:
            self.init_doc_states(docs)
            self.sample_doc_states(docs, save_dir, iters, iters_lag)

    def init_doc_states(self, docs):
        for doc in docs:
            # doc.state = self.sample_doc_state(doc)

            state_log_likes = np.zeros(self.num_states)
            for s in range(self.num_states):
                state_log_likes[s] = self.init_state_log_like(doc, s) + \
                                     self.calc_state_topic_log_like_arr(doc, s)
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
            "alphas": self.alphas.tolist(),
            "num_sequences": self.num_sequences,
        }
        json_str = json.dumps(params)
        append_to_file(os.path.join(save_dir, OUT_PARAMS), [ts, i, json_str])

        # data structures capturing results of latest sampling iteration
        json_str = json.dumps(self.state_trans)
        append_to_file(os.path.join(save_dir, OUT_STATE_TRANS), [ts, i, json_str])

        json_str = json.dumps([ s.tolist() for s in self.state_topic_counts ])
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

    def delete_progress(self, save_dir):
        del_count = 0
        for fname in [OUT_PARAMS, OUT_STATE_TRANS, OUT_STATE_TOPICS, OUT_STATES]:
            try:
                path = os.path.join(save_dir, fname)
                os.remove(path)
                sys.stderr.write("removed " + path + "\n")
                del_count += 1
            except OSError:
                continue
        return del_count

    def sample_doc_states(self, docs, save_dir, iterations, lag_iters, start_iter=0):
        timing_iters = 10  # calc a moving average of time per iter over this many
        timing_start = datetime.datetime.now()

        for i in range(start_iter, iterations):
            logging.debug("iter {}".format(i))
            if i % timing_iters == 0:
                ts_now = datetime.datetime.now()
                logging.debug("current pace {}/iter".format((ts_now - timing_start)//timing_iters))
                timing_start = ts_now

            for d, doc in enumerate(docs):
                if d % 500 == 0:
                    logging.debug("iter {}, doc {}".format(i, d))

                self.remove_from_trans_counts(doc)
                self.remove_from_topic_counts(doc)

                # # Old way, no parallelism
                # state_log_likes = np.zeros(self.num_states)
                # # todo: handle these with array funcs rather than iterating over states
                # for s in range(self.num_states):
                #     state_log_likes[s] = self.calc_state_state_log_like(doc, s)
                #     state_log_likes[s] += self.calc_state_topic_log_like_arr(doc, s)

                # # Using joblib naively
                # state_log_likes = Parallel(n_jobs=2)(delayed(calc_state_log_like)(self, doc, s) for s in range(self.num_states))

                # Using joblib and extracted function
                with Parallel(n_jobs=4) as parallel:
                    state_log_likes = parallel(delayed(calc_state_log_like)(
                                                                self.alphas,
                                                                self.state_topic_counts[s],
                                                                self.sum_alpha,
                                                                self.state_topic_totals[s],
                                                                self.pi,
                                                                self.start_counts[s],
                                                                self.num_sequences,
                                                                self.sum_pi,
                                                                s,
                                                                self.gamma,
                                                                self.state_trans_tots[s],
                                                                self.sum_gamma,
                        state_trans_prev=self.state_trans[doc.doc_prev.state][s] if
                            doc.doc_prev is not None else None,
                        state_trans_next=self.state_trans[s][doc.doc_next.state] if
                            doc.doc_next is not None else None,
                        doc_len=doc.length,
                        doc_prev_state=doc.doc_prev.state if
                            doc.doc_prev is not None else None,
                        doc_next_state=doc.doc_next.state if
                            doc.doc_next is not None else None,
                        topic_distrib=doc.topic_distrib
                    ) for s in range(self.num_states))

                # # Using extracted function, no parallel
                # state_log_likes = np.zeros(self.num_states)
                # for s in range(self.num_states):
                #     state_log_likes[s] = calc_state_log_like(self.alphas,
                #                                                 self.state_topic_counts[s],
                #                                                 doc,
                #                                                 self.sum_alpha,
                #                                                 self.state_topic_totals[s],
                #                                                 self.pi,
                #                                                 self.start_counts[s],
                #                                                 self.num_sequences,
                #                                                 self.sum_pi,
                #                                                 self.state_trans,
                #                                                 s,
                #                                                 self.gamma,
                #                                                 self.state_trans_tots[s],
                #                                                 self.sum_gamma)

                # # Using multiprocessing pool
                # pool = Pool(processes=4)
                # arg_lists = [ (
                #                                                 self.alphas,
                #                                                 self.state_topic_counts[s],
                #                                                 doc,
                #                                                 self.sum_alpha,
                #                                                 self.state_topic_totals[s],
                #                                                 self.pi,
                #                                                 self.start_counts[s],
                #                                                 self.num_sequences,
                #                                                 self.sum_pi,
                #                                                 self.state_trans,
                #                                                 s,
                #                                                 self.gamma,
                #                                                 self.state_trans_tots[s],
                #                                                 self.sum_gamma
                #     ) for s in range(self.num_states)]
                # state_log_likes = pool.map(calc_state_log_like, arg_lists)



                doc.state = sample_from_loglikes(state_log_likes)
                self.add_to_trans_counts(doc)
                self.add_to_topic_counts(doc)

            if i % lag_iters == 0:
                self.save_progress(i, docs, save_dir)

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
            # trace += "begin "
            lik = (self.start_counts[s] + self.pi) / (self.num_sequences - 1 + self.sum_pi)
            # trace += "({} + {}) / ({} - 1 + {}) = {}".format(self.start_counts[s], self.pi,
            #                                                  self.num_sequences, self.sum_pi,
            #                                                  lik)
            if doc.doc_next is not None:  # not a singleton sequence
                # trace += "cont "
                # print "state ", s, " doc_next.state ", doc.doc_next.state
                lik *= self.state_trans[s][doc.doc_next.state] + self.gamma

        else:  # middle of sequence
            # trace += "middle "
            if doc.doc_next is None:  # end of sequence
                lik = (self.state_trans[doc.doc_prev.state][s] + self.gamma)
                # trace += "end "
            else:
                # trace += "cont "
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

        # print "lik for state ", s, "(", trace, ")", ": ", lik
        return math.log(lik)

    def calc_state_topic_log_like_arr(self, doc, state):
        ret = 0.0

        den = self.alphas + self.state_topic_counts[state]
        num = den + doc.topic_distrib
        ret += np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den))

        ret += math.lgamma(self.sum_alpha + self.state_topic_totals[state]) - \
            math.lgamma(self.sum_alpha + self.state_topic_totals[state] + doc.length)

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

    def add_to_topic_counts(self, doc):
        # for topic, topic_count in enumerate(doc.topic_distrib):
        #     self.state_topic_counts[doc.state][topic] += topic_count
        # self.state_topic_totals[doc.state] += doc.length
        self.state_topic_counts[doc.state] += doc.topic_distrib
        self.state_topic_totals[doc.state] += doc.length

    def remove_from_topic_counts(self, doc):
        # for topic, topic_count in enumerate(doc.topic_distrib):
        #     if topic_count > 0:
        #         self.state_topic_counts[doc.state][topic] -= topic_count
        # self.state_topic_totals[doc.state] -= doc.length
        self.state_topic_counts[doc.state] -= doc.topic_distrib
        self.state_topic_totals[doc.state] -= doc.length

class Document(object):
    def __init__(self, res_ent, topic_distrib, doc_prev=None, doc_next=None):
        self.entry = res_ent
        self.topic_distrib = np.array(topic_distrib)
        self.doc_prev = doc_prev
        self.doc_next = doc_next

        self.length = sum(self.topic_distrib)
        self.state = None


def sample_from_loglikes(state_log_likes):
    # turn log likes into a distrib to sample from
    # state_log_like_max = max(state_log_likes)
    # state_likes_divmax = [ math.exp(loglik - state_log_like_max) for loglik in state_log_likes ]
    # norm = sum(state_likes_divmax)
    # state_samp_distrib = [ lik/norm for lik in state_likes_divmax ]
    # state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
    state_log_like_max = np.max(state_log_likes)
    state_likes_divmax = np.exp(state_log_likes - state_log_like_max)
    norm = np.sum(state_likes_divmax)
    state_samp_distrib = state_likes_divmax/norm
    state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
    return state_new


# wrapper to make it work with joblib
def calc_state_log_like(alphas,
                        state_top_counts,
                        sum_alpha,
                        state_top_total,
                        pi,
                        start_count,
                        num_seqs,
                        sum_pi,
                        state,
                        gamma,
                        state_trans_tot,
                        sum_gamma,
                        state_trans_prev,
                        state_trans_next,
                        doc_len,
                        doc_prev_state,
                        doc_next_state,
                        topic_distrib
                        ):
# return hmm.calc_state_state_log_like(doc, state) + hmm.calc_state_topic_log_like_arr(doc, state)
# def calc_state_log_like(args):
#     alphas, state_top_counts, doc, sum_alpha, state_top_total, pi, start_count, num_seqs, \
#         sum_pi, state_trans, state,gamma, state_trans_tot, sum_gamma = args

    ret = 0.0

    # state-topic log like
    den = alphas + state_top_counts
    num = den + topic_distrib
    ret += np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den))

    ret += math.lgamma(sum_alpha + state_top_total) - \
            math.lgamma(sum_alpha + state_top_total + doc_len)

    # state-state log like
    if state_trans_prev is None:  # beginning of resume sequence
        lik = (start_count + pi) / (num_seqs - 1 + sum_pi)
        if state_trans_next is not None:  # not a singleton sequence
            lik *= state_trans_next + gamma

    else:  # middle of sequence
        if state_trans_next is None:  # end of sequence
            lik = (state_trans_prev + gamma)
            # trace += "end "
        else:
            # trace += "cont "
            if (doc_prev_state == state) and (state == doc_next_state):
                lik = ((state_trans_prev + gamma) *
                       (state_trans_next + 1 + gamma) /
                       (state_trans_tot + 1 + sum_gamma))

            elif doc_prev_state == state:  # and (s != doc_next.state):
                lik = ((state_trans_prev + gamma) *
                       (state_trans_next + gamma) /
                       (state_trans_tot + 1 + sum_gamma))

            else:  # (doc_prev.state != s)
                lik = ((state_trans_prev + gamma) *
                       (state_trans_next + gamma) /
                       (state_trans_tot + sum_gamma))
    ret += math.log(lik)

    return ret


def get_docs_from_resumes(resume_list, min_len=1):
    docs = []

    debug_len_distrib = np.zeros(20, int)

    for r, resume in enumerate(resume_list):
        resume_len = len(resume)
        debug_len_distrib[min(19, resume_len)] += 1
        if r % 10000 == 0:
            logging.debug("\t{} {}".format(r, debug_len_distrib))

        if resume_len < min_len:
            pass
        elif resume_len == 1:
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

    logging.debug("\t{} {}".format(r, debug_len_distrib))
    return docs


def append_to_file(file_name, elts):
    with open(file_name, 'a') as out:
        out.write("\t".join([ str(e) for e in elts ]) + "\n")


def read_last_line(file_name):
    line = None
    with open(file_name, 'r') as infile:
        for line in infile:
            pass
        return line.rstrip("\n").split("\t")


def debug_audit_state_trans_tots(docs):
    num_states = max([ doc.state for doc in docs ]) + 1
    state_trans_tots = [0] * num_states
    for doc in docs:
        if doc.doc_next is not None:
            state_trans_tots[doc.state] += 1
    return state_trans_tots


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
    parser.add_argument('--erase', action='store_true')

    args = parser.parse_args()

    # get a list of lists of (ResumeEntry, topic_distrib) tuples
    logging.info("loading resumes from file")
    resumes = load_json_resumes_lda(args.infile)
    num_tops = len(resumes[0][0][1])  # distrib for the first job entry in the first resume
    logging.info("extracting documents from resumes")
    resume_docs = get_docs_from_resumes(resumes)

    logging.info("fitting HMM")
    hmm = ResumeHMM(args.num_states, args.pi, args.gamma, num_tops)
    hmm.fit(resume_docs, args.savedir, args.num_iters, args.lag, erase=args.erase)

    print "yo zzz"