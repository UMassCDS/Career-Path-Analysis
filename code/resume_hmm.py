import sys
import math
import argparse
import json
import datetime
import logging
import os
# import ctypes
import os.path
import numpy as np
import multiprocessing
import scipy.special

# import resume_common
from resume_lda import load_json_resumes_lda, scan_json_resumes_lda


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


OUT_PARAMS = 'params.tsv'
OUT_START_COUNTS = 'starts.tsv'
OUT_STATE_TRANS = 'trans.tsv'
OUT_STATE_TOPICS = 'topics.tsv'
OUT_STATES = 'states.tsv'

NULL_DOC = -1  # we use c-type arrays to capture prev and next docs, so use this in place of None

# these are declared locally so they'll be available to subprocesses for free
pi = 0.0        # (uniform) prior on start state
gamma = 0.0     # (uniform) prior on state-state transitions
sum_pi = 0.0
sum_gamma = 0.0
alphas = []     # prior on topic distrib
sum_alpha = 0.0

# num_states
# num_topics
# alphas                1xS
# pi
# gamma
# sum_alpha
# sum_pi
# sum_gamma

# start_counts          1xS
# state_trans           SxS
# state_trans_tots      1xS
# state_topic_counts    TxS
# state_topic_totals    1xS

# num_sequences
# documents             1xD
# document_states       1xD


class ResumeHmm(object):
    def __init__(self, p_num_states, p_pi, p_gamma, p_num_topics):
        global pi, gamma, sum_pi, sum_gamma, alphas, sum_alpha
        pi = p_pi        # (uniform) prior on start state
        gamma = p_gamma  # (uniform) prior on state-state transitions
        sum_pi = p_pi*p_num_states
        sum_gamma = p_gamma*p_num_states

        self.num_states = p_num_states
        self.num_topics = p_num_topics

        sum_alpha = self.num_topics
        alphas = np.array([sum_alpha/self.num_topics]*self.num_topics, np.double)

    def fit(self, save_dir, iters, iters_lag, pool, erase=False):
        if erase:
            self.delete_progress(save_dir)
        if os.path.isfile(os.path.join(save_dir, OUT_PARAMS)):
            i = self.load_progress(save_dir)
            self.sample_doc_states(save_dir, iters, iters_lag, pool, start_iter=i+1)
        else:
            self.state_topic_counts = np.zeros((self.num_states, self.num_topics), np.double)
            self.state_topic_totals = np.zeros(self.num_states, np.double)

            self.start_counts = np.zeros(self.num_states, np.int)
            self.state_trans = np.zeros((self.num_states, self.num_states), np.int)
            self.state_trans_tots = np.zeros(self.num_states, np.int)

            self.init_doc_states(pool)

            self.sample_doc_states(save_dir, iters, iters_lag, pool)

    def init_doc_states(self, pool):
        logging.debug("initializing states for {} documents".format(self.num_docs))

        # pool = multiprocessing.Pool(processes=num_procs)

        for d in range(self.num_docs):
            if d % 50000 == 0:
                logging.debug("initializing doc {}/{}".format(d, self.num_docs - 1))

            args = [(self.state_topic_counts[s],
                     self.state_topic_totals[s],
                     self.doc_topic_distribs[d],
                     self.doc_lens[d],
                     self.start_counts[s],
                     self.num_sequences,
                     self.state_trans[self.doc_states[self.doc_prevs[d]], s] if
                                                        self.doc_prevs[d] != NULL_DOC else None
                    ) for s in range(self.num_states)]

            if pool is None:
                state_log_likes = [init_state_log_like(a) for a in args]
            else:
                state_log_likes = pool.map(init_state_log_like, args)
            new_state = sample_from_loglikes(state_log_likes)
            self.doc_states[d] = new_state

            self.init_trans_counts(d)
            self.add_to_topic_counts(d)

        # pool.terminate()

    def init_trans_counts(self, d):
        doc_state = self.doc_states[d]
        doc_prev = self.doc_prevs[d]
        doc_next = self.doc_nexts[d]
        if doc_prev == NULL_DOC:  # beginning of resume sequence
            self.start_counts[doc_state] += 1
        else:  # middle of sequence
            self.state_trans[self.doc_states[doc_prev], doc_state] += 1

        if doc_next != NULL_DOC:  # not the end of sequence
            self.state_trans_tots[doc_state] += 1

    # def sample_doc_states(self, save_dir, iterations, lag_iters, start_iter=0, num_procs=1):
    def sample_doc_states(self, save_dir, iterations, lag_iters, pool, start_iter=0):
        timing_iters = 1  # calc a moving average of time per iter over this many
        timing_start = datetime.datetime.now()

        # # create a multiprocessing pool that can be reused each iteration
        # pool = multiprocessing.Pool(processes=num_procs)
        #
        for i in range(start_iter, iterations):
            logging.debug("iter {}".format(i))
            if i % timing_iters == 0:
                ts_now = datetime.datetime.now()
                logging.debug("current pace {}/iter".format((ts_now - timing_start)//timing_iters))
                timing_start = ts_now

            # for d, doc in enumerate(docs):
            for d in range(self.num_docs):
                if d % 50000 == 0:
                    logging.debug("iter {}/{}, doc {}/{}".format(i, iterations-1, d, self.num_docs-1))

                self.remove_from_trans_counts(d)
                self.remove_from_topic_counts(d)

                # if num_procs == 1:
                #     state_log_likes = [ calc_state_log_like((d, s)) for s in range(num_states) ]
                # else:
                #     args = [ (d, s) for s in range(num_states) ]
                #     state_log_likes = pool.map(calc_state_log_like, args)

                # args = [ (self.state_topic_counts[s],
                #           self.state_topic_totals[s],
                #
                #           self.doc_topic_distribs[d],
                #           self.doc_lens[d],
                #
                #           self.start_counts[s], self.num_sequences, s, self.doc_prevs[d], self.doc_nexts[d],
                #
                #           self.state_trans[self.doc_states[self.doc_prevs[d]], s] if
                #                                             self.doc_prevs[d] != NULL_DOC else None,
                #           self.state_trans[self.doc_states[self.doc_nexts[d]], s] if
                #                                             self.doc_nexts[d] != NULL_DOC else None,
                #           self.state_trans_tots[s]) for s in range(self.num_states) ]
                #
                # if pool is None:
                #     state_log_likes = [ calc_state_log_like(a) for a in args ]
                # else:
                #     state_log_likes = pool.map(calc_state_log_like, args)
                #
                #

                # state_log_likes2a = calc_state_topic_log_like_matrix(alphas, sum_alpha,
                #                                                     self.state_topic_counts,
                #                                                     self.state_topic_totals,
                #                                                     self.doc_topic_distribs[d],
                #                                                     self.doc_lens[d])
                state_log_likes2a = self.calc_state_topic_log_like_matrix(d)

                doc_prev = self.doc_prevs[d]
                doc_prev_state = self.doc_states[doc_prev] if doc_prev != NULL_DOC else None
                doc_next = self.doc_nexts[d]
                doc_next_state = self.doc_states[doc_next] if doc_next != NULL_DOC else None
                state_log_likes2b = calc_state_state_log_like_matrix(self.doc_states[d],
                                                                     doc_prev_state,
                                                                     doc_next_state,
                                                                     self.state_trans,
                                                                     self.state_trans_tots,
                                                                     pi, sum_pi, gamma, sum_gamma,
                                                                     self.start_counts,
                                                                     self.num_sequences)
                state_log_likes = state_log_likes2a + state_log_likes2b

                # print "doc {}, iter {}:".format(d, i)
                # print state_log_likes[:5]
                # print state_log_likes2[:5]
                # print "\n\n"

                self.doc_states[d] = sample_from_loglikes(state_log_likes)
                self.add_to_trans_counts(d)
                self.add_to_topic_counts(d)

            if i % lag_iters == 0:
                self.save_progress(i, save_dir)

                # pool.terminate()

    def calc_state_topic_log_like_matrix(self, d):

        # state_topic_counts is (s x t), so each state is a row, each topic a col
        den = self.state_topic_counts + alphas  # SxT
        num = den + self.doc_topic_distribs[d]  # SxT
        state_sums = np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den), axis=1)  # Sx1

        num = self.state_topic_totals + sum_alpha  # Sx1
        den = num + self.doc_lens[d]  # Sx1
        state_sums += scipy.special.gammaln(num) - scipy.special.gammaln(den)  # Sx1

        return state_sums

    def add_to_trans_counts(self, d):
        doc_state = self.doc_states[d]
        doc_prev = self.doc_prevs[d]
        doc_next = self.doc_nexts[d]

        if doc_prev == NULL_DOC:  # beginning of resume sequence
            self.start_counts[doc_state] += 1
            if doc_next != NULL_DOC:  # not a singleton sequence
                self.state_trans[doc_state, self.doc_states[doc_next]] += 1
                self.state_trans_tots[doc_state] += 1
        else:  # middle of sequence
            self.state_trans[self.doc_states[doc_prev], doc_state] += 1
            if doc_next != NULL_DOC:  # not the end of sequence
                self.state_trans[doc_state, self.doc_states[doc_next]] += 1
                self.state_trans_tots[doc_state] += 1

    def remove_from_trans_counts(self, d):
        doc_state = self.doc_states[d]
        doc_prev = self.doc_prevs[d]
        doc_next = self.doc_nexts[d]

        if doc_prev == NULL_DOC:  # beginning of resume sequence
            self.start_counts[doc_state] -= 1
            if doc_next != NULL_DOC:  # not a singleton sequence
                self.state_trans[doc_state, self.doc_states[doc_next]] -= 1
                self.state_trans_tots[doc_state] -= 1
        else:  # middle of sequence
            self.state_trans[self.doc_states[doc_prev], doc_state] -= 1
            if doc_next != NULL_DOC:  # not end of sequence
                self.state_trans[doc_state, self.doc_states[doc_next]] -= 1
                self.state_trans_tots[doc_state] -= 1

    def add_to_topic_counts(self, d):
        doc_state = self.doc_states[d]
        self.state_topic_counts[doc_state] += self.doc_topic_distribs[d]
        self.state_topic_totals[doc_state] += self.doc_lens[d]

    def remove_from_topic_counts(self, d):
        doc_state = self.doc_states[d]
        self.state_topic_counts[doc_state] -= self.doc_topic_distribs[d]
        self.state_topic_totals[doc_state] -= self.doc_lens[d]

    def load_docs_from_resumes(self, infile_name, min_len=1, max_docs=sys.maxint):

        resumes = load_json_resumes_lda(infile_name, max_docs)

        # global document_topic_distribs, document_prevs, document_nexts, document_lens, document_states
        # global num_documents, num_sequences
        # global document_topic_distribs_base

        debug_len_distrib = np.zeros(20, np.int_)
        # docs = []
        # doc_idx = 0
        # for r, resume in enumerate(resume_list):
        #     resume_len = len(resume)
        #     debug_len_distrib[min(19, resume_len)] += 1
        #     if r % 10000 == 0:
        #         logging.debug("\t{} {}".format(r, debug_len_distrib))
        #
        #     if resume_len < min_len:
        #         continue
        #
        #     idxs = range(doc_idx, doc_idx + resume_len)
        #     prevs = [None] + idxs[:-1]
        #     nexts = idxs[1:] + [None]
        #     for i, (res_ent, top_dis) in enumerate(resume):
        #         docs.append(Document(prevs[i], nexts[i], top_dis))
        #
        #     doc_idx += resume_len

        # we already called this, so technically we should pass it in
        num_seqs, num_docs, num_topics = scan_json_resumes_lda(args.infile)

        self.doc_prevs = np.ndarray(num_docs, np.int)
        self.doc_nexts = np.ndarray(num_docs, np.int)
        self.doc_topic_distribs = np.ndarray((num_docs, num_topics))
        self.doc_lens = np.ndarray(num_docs, np.int)

        doc_idx = 0
        res_count = 0
        for r, resume in enumerate(resumes):
            resume_len = len(resume)
            debug_len_distrib[min(19, resume_len)] += 1
            if r % 10000 == 0:
                logging.debug("\t{} {}".format(r, debug_len_distrib))

            if resume_len < min_len:
                continue

            idxs = range(doc_idx, doc_idx + resume_len)
            prevs = [NULL_DOC] + idxs[:-1]  # if it's missing,
            nexts = idxs[1:] + [NULL_DOC]

            # this could be done faster with array copy stuff
            for i, (res_ent, top_dis) in enumerate(resume):
                self.doc_prevs[doc_idx + i] = prevs[i]
                self.doc_nexts[doc_idx + i] = nexts[i]
                self.doc_topic_distribs[doc_idx + i] = top_dis
                self.doc_lens[doc_idx + i] = sum(top_dis)

            doc_idx += resume_len
            res_count += 1

            if doc_idx >= max_docs:
                break


        # now globalize 'em
        # logging.debug("globalizing document arrays")
        # document_prevs_base = multiprocessing.Array(ctypes.c_int, doc_prevs)
        # document_prevs = wrap_global_array(document_prevs_base, doc_idx)
        #
        # document_nexts_base = multiprocessing.Array(ctypes.c_int, doc_nexts)
        # document_nexts = wrap_global_array(document_nexts_base, doc_idx)

        # See: https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing
        # document_topic_distribs_base = multiprocessing.Array(ctypes.c_double,
        #                                                      np.ravel(doc_topic_distribs))
        # document_topic_distribs = wrap_global_array(document_topic_distribs_base, (doc_idx, -1))

        # document_topic_distribs = np.array(doc_topic_distribs, np.double)
        #
        # # document_prevs = multiprocessing.Array('i', doc_prevs)
        # # document_nexts = multiprocessing.Array('i', doc_nexts)
        # # document_lens = multiprocessing.Array(ctypes.c_double, doc_lens)
        # # document_states = multiprocessing.Array('i', [-1]*doc_idx)
        # document_prevs = np.array(doc_prevs, np.int)
        # document_nexts = np.array(doc_nexts, np.int)
        # document_lens = np.array(doc_lens, np.int)
        # document_states = np.array([-1] * doc_idx, np.int)
        #
        self.num_docs = doc_idx
        self.num_sequences = res_count
        self.doc_states = np.ndarray(self.num_docs, np.int)

    def save_progress(self, i, save_dir):
        ts = str(datetime.datetime.now())

        # model parameters (should not change between iters)
        params = {
            "num_states": self.num_states,
            "pi": pi,
            "gamma": gamma,
            "alphas": [a for a in alphas],
            "num_sequences": self.num_sequences,
        }
        json_str = json.dumps(params)
        append_to_file(os.path.join(save_dir, OUT_PARAMS), [ts, i, json_str])

        # data structures capturing results of latest sampling iteration
        # zzz todo fix!
        # json_str = json.dumps([c for c in start_counts])
        # append_to_file(os.path.join(save_dir, OUT_START_COUNTS), [ts, i, json_str])

        json_str = json.dumps(self.state_trans.tolist())
        append_to_file(os.path.join(save_dir, OUT_STATE_TRANS), [ts, i, json_str])

        json_str = json.dumps([s.tolist() for s in self.state_topic_counts])
        append_to_file(os.path.join(save_dir, OUT_STATE_TOPICS), [ts, i, json_str])

        json_str = json.dumps([s for s in self.doc_states])
        append_to_file(os.path.join(save_dir, OUT_STATES), [ts, i, json_str])

    def load_progress(self, save_dir):
        ts, iter_params, json_str = read_last_line(os.path.join(save_dir, OUT_PARAMS))
        params = json.loads(json_str)
        self.num_states = params["num_states"]
        pi = params["pi"]
        gamma = params["gamma"]
        alphas = params["alphas"]
        self.num_sequences = params["num_sequences"]

        ts, iter_starts, json_str = read_last_line(os.path.join(save_dir, OUT_START_COUNTS))
        self.start_counts = np.array(json.loads(json_str), np.int_)

        ts, iter_trans, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TRANS))
        # zzz todo: this is broken
        self.state_trans = np.array(json.loads(json_str), np.int_)

        ts, iter_topics, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TOPICS))
        # zzz todo: this is broken
        self.state_topic_counts = np.array(json.loads(json_str), np.double)

        ts, iter_states, json_str = read_last_line(os.path.join(save_dir, OUT_STATES))
        doc_states = json.loads(json_str)
        # for doc, state in zip(documents, doc_states):
        #     doc.state = state
        for i, s in enumerate(doc_states):
            self.doc_states[i] = s

        if iter_params == iter_starts == iter_trans == iter_topics == iter_states:
            return iter_params
        else:
            sys.exit("unequal iter counts loaded")

    def delete_progress(self, save_dir):
        del_count = 0
        for fname in [OUT_PARAMS, OUT_START_COUNTS, OUT_STATE_TRANS, OUT_STATE_TOPICS, OUT_STATES]:
            try:
                path = os.path.join(save_dir, fname)
                os.remove(path)
                sys.stderr.write("removed " + path + "\n")
                del_count += 1
            except OSError:
                continue
        return del_count






# def calc_state_log_like(params):
#     d, start, end = params
#     rets = []
#     for s in range(start, end):
#         ret = calc_state_topic_log_like(d, s)
#         ret += calc_state_state_log_like(d, s)
#         rets.append(ret)
#     return rets


# def calc_state_log_like(param_chunk):
#     rets = []
#     for s, d in param_chunk:
#         ret = calc_state_topic_log_like(s, d)
#         ret += calc_state_state_log_like(s, d)
#         rets.append(ret)
#     return rets
#

def init_state_log_like(params):
    topic_counts, topic_total, topic_distrib, doc_len, start_count, num_sequences, state_trans_prev = params

    if state_trans_prev is None:
        lik = (start_count + pi) / (num_sequences - 1 + sum_pi)
    else:
        lik = state_trans_prev + gamma
    loglik = math.log(lik)

    loglik += calc_state_topic_log_like(topic_counts, topic_total, topic_distrib, doc_len)
    return loglik


def calc_state_log_like(params):
    topic_counts, topic_total, topic_distrib, doc_len, \
    start_count, num_seqs, s, state_prev, state_next, \
    state_trans_prev, state_trans_next, state_trans_tot = params

    ret = 0.0
    ret += calc_state_topic_log_like(topic_counts, topic_total, topic_distrib, doc_len)
    ret += calc_state_state_log_like(start_count, num_seqs, s, state_prev, state_next,
                              state_trans_prev, state_trans_next, state_trans_tot)
    return ret


def calc_state_topic_log_like(topic_counts, topic_total, topic_distrib, doc_len):
    ret = 0.0

    den = alphas + topic_counts
    num = den + topic_distrib
    ret += np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den))

    ret += math.lgamma(sum_alpha + topic_total) - \
           math.lgamma(sum_alpha + topic_total + doc_len)
    return ret

def calc_state_topic_log_like_matrix(alphas, sum_alpha,
                                     state_topic_counts, state_topic_totals,
                                     doc_topic_distrib, doc_len):
    # state_topic_counts is (s x t), so each state is a row, each topic a col
    den = state_topic_counts + alphas  # s x t
    num = den + doc_topic_distrib      # s x t
    state_sums = np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den), axis=1)  # s x 1

    num = state_topic_totals + sum_alpha  # s x 1
    den = num + doc_len                   # s x 1
    state_sums += scipy.special.gammaln(num) - scipy.special.gammaln(den)  # s x 1

    return state_sums


def calc_state_state_log_like(start_count, num_sequences, s, state_prev, state_next,
                              state_trans_prev, state_trans_next, state_trans_tot):

    if state_trans_prev is None:  # beginning of resume sequence
        lik = (start_count + pi) / (num_sequences - 1 + sum_pi)
        if state_trans_next is not None:  # not a singleton sequence
            lik *= state_trans_next + gamma

    else:  # middle of sequence
        if state_trans_next is None:  # end of sequence
            lik = state_trans_prev + gamma
        else:
            if state_prev == s:
                if s == state_next:
                    lik = ((state_trans_prev + gamma) *
                           (state_trans_next + 1 + gamma) /
                           (state_trans_tot + 1 + sum_gamma))
                else:
                    lik = ((state_trans_prev + gamma) *
                           (state_trans_next + gamma) /
                           (state_trans_tot + 1 + sum_gamma))
            else:  # (doc_prev.state != s)
                lik = ((state_trans_prev + gamma) *
                       (state_trans_next + gamma) /
                       (state_trans_tot + sum_gamma))
    # print "lik for state ", s, "(", trace, ")", ": ", lik
    return math.log(lik)


def calc_state_state_log_like_matrix(doc_state, doc_prev_state, doc_next_state,
                                     state_trans, state_trans_tots,
                                     pi, sum_pi, gamma, sum_gamma,
                                     state_start_counts, num_seqs):

    if doc_prev_state is None:  # beginning of resume sequence
        state_liks = state_start_counts + pi  # s x 1
        state_liks /= (num_seqs - 1 + sum_pi)

        if doc_next_state is not None:  # not a singleton sequence
            state_liks *= state_trans[:, doc_next_state] + gamma

    else:  # middle of sequence
        if doc_next_state is None:  # end of sequence
            state_liks = state_trans[doc_prev_state, :] + gamma
        else:
            if doc_prev_state == doc_state:
                if doc_state == doc_next_state:
                    state_liks = state_trans[doc_prev_state, :] + gamma
                    state_liks *= state_trans[:, doc_next_state] + 1 + gamma
                    state_liks /= state_trans_tots + 1 + sum_gamma
                else:
                    state_liks = state_trans[doc_prev_state, :] + gamma
                    state_liks *= state_trans[:, doc_next_state] + gamma
                    state_liks /= state_trans_tots + 1 + sum_gamma
            else:  # (doc_prev.state != s)
                state_liks = state_trans[doc_prev_state, :] + gamma
                state_liks *= state_trans[:, doc_next_state] + gamma
                state_liks /= state_trans_tots + sum_gamma

    return np.log(state_liks)


def sample_from_loglikes(state_log_likes):
    # turn log likes into a distrib from which to sample
    state_log_like_max = np.max(state_log_likes)
    state_likes_divmax = np.exp(state_log_likes - state_log_like_max)
    norm = np.sum(state_likes_divmax)
    state_samp_distrib = state_likes_divmax/norm
    state_new = np.random.choice(len(state_samp_distrib), p=state_samp_distrib)
    return state_new


# class Document(object):
#     def __init__(self, doc_prev, doc_next, topic_distrib):
#         self.doc_prev = doc_prev
#         self.doc_next = doc_next
#         self.topic_distrib = np.array(topic_distrib)
#         self.length = sum(topic_distrib)
#         # store these in a global array instead to make multiprocessing happy?
#         # self.state = None


# See: https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing
# shared_array_base = multiprocessing.Array(ctypes.c_double, 10 * 10)
# shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
# shared_array = shared_array.reshape(10, 10)
def wrap_global_array(shared_array_base, shape):
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    return shared_array.reshape(shape)



def append_to_file(file_name, elts):
    with open(file_name, 'a') as out:
        out.write("\t".join([ str(e) for e in elts ]) + "\n")


def read_last_line(file_name):
    line = None
    with open(file_name, 'r') as infile:
        for line in infile:
            pass
        return line.rstrip("\n").split("\t")


# def debug_audit_state_trans_tots(docs):
#     # num_states = max([ doc.state for doc in docs ]) + 1
#     num_states = max(document_states) + 1
#     state_trans_tots = [0] * num_states
#     for d, doc in enumerate(docs):
#         if doc.doc_next != NULL_DOC:
#             state_trans_tots[document_states[d]] += 1
#     return state_trans_tots


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
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--num_docs', type=int, default=sys.maxint)

    args = parser.parse_args()

    logging.info("scanning resume file")
    num_seqs, num_docs, num_tops = scan_json_resumes_lda(args.infile, args.min_len, args.num_docs)
    logging.info("{} resumes, {} jobs, {} topics".format(num_seqs, num_docs, num_tops))

    hmm = ResumeHmm(args.num_states, args.pi, args.gamma, num_tops)

    # create a multiprocessing pool that can be reused each iteration
    if args.num_procs > 1:
        logging.info("allocating {} subprocesses".format(args.num_procs))
        pool = multiprocessing.Pool(processes=args.num_procs)
    else:
        pool = None

    # get a list of lists of (ResumeEntry, topic_distrib) tuples
    logging.info("loading resumes from file")
    hmm.load_docs_from_resumes(args.infile, min_len=args.min_len, max_docs=args.num_docs)

    # resumes = load_json_resumes_lda()
    #
    # logging.info("extracting documents from resumes")
    # get_docs_from_resumes(resumes)
    # del resumes

    logging.info("fitting HMM")
    hmm.fit(args.savedir, args.num_iters, args.lag, pool, erase=args.erase)

    if pool is not None:
        logging.info("killing subprocesses")
        pool.terminate()

    print "yo zzz"