import sys
import argparse
import json
import datetime
import logging
import os
import os.path
import numpy as np
import scipy.special
from resume_lda import load_json_resumes_lda, scan_json_resumes_lda


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


OUT_PARAMS = 'params.tsv'
OUT_START_COUNTS = 'starts.tsv'
OUT_STATE_TRANS = 'trans.tsv'
OUT_STATE_TOPICS = 'topics.tsv'
OUT_STATES = 'states.tsv'
NULL_DOC = -1  # we use numpy arrays to capture prev and next docs, so use this in place of None

# num_states
# num_topics
# alphas                Sx1
# pi
# gamma
# sum_alpha
# sum_pi
# sum_gamma
# start_counts          Sx1
# state_trans           SxS
# state_trans_tots      Sx1
# state_topic_counts    TxS
# state_topic_totals    Sx1
# num_sequences
# documents             Dx1
# document_states       Dx1


class ResumeHmm(object):
    def __init__(self, p_num_states, p_pi, p_gamma, p_num_topics):
        self.pi = p_pi        # (uniform) prior on start state
        self.gamma = p_gamma  # (uniform) prior on state-state transitions
        self.sum_pi = p_pi*p_num_states
        self.sum_gamma = p_gamma*p_num_states

        self.num_states = p_num_states
        self.num_topics = p_num_topics

        # prior on topic distribs
        self.sum_alpha = self.num_topics
        self.alphas = np.array([self.sum_alpha/self.num_topics]*self.num_topics, np.double)  # Sx1

    def fit(self, save_dir, iters, iters_lag, erase=False):
        if erase:
            self.delete_progress(save_dir)
        if os.path.isfile(os.path.join(save_dir, OUT_PARAMS)):
            i = self.load_progress(save_dir)
            self.sample_doc_states(save_dir, iters, iters_lag, start_iter=i+1)
        else:
            self.state_topic_counts = np.zeros((self.num_states, self.num_topics), np.double)
            self.state_topic_totals = np.zeros(self.num_states, np.double)

            self.start_counts = np.zeros(self.num_states, np.int)
            self.state_trans = np.zeros((self.num_states, self.num_states), np.int)
            self.state_trans_tots = np.zeros(self.num_states, np.int)

            self.init_doc_states()

            self.sample_doc_states(save_dir, iters, iters_lag)

    def init_doc_states(self):
        logging.debug("initializing states for {} documents".format(self.num_docs))

        for d in range(self.num_docs):
            if d % 50000 == 0:
                logging.debug("initializing doc {}/{}".format(d, self.num_docs - 1))

            state_log_likes = self.init_state_log_like_matrix(d)

            new_state = sample_from_loglikes(state_log_likes)
            self.doc_states[d] = new_state

            self.init_trans_counts(d)
            self.add_to_topic_counts(d)

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

    def sample_doc_states(self, save_dir, iterations, lag_iters, start_iter=0):
        timing_iters = 1  # calc a moving average of time per iter over this many
        timing_start = datetime.datetime.now()

        for i in range(start_iter, iterations):
            logging.debug("iter {}".format(i))
            if i % timing_iters == 0:
                ts_now = datetime.datetime.now()
                logging.debug("current pace {}/iter".format((ts_now - timing_start)//timing_iters))
                timing_start = ts_now

            for d in range(self.num_docs):
                if d % 50000 == 0:
                    logging.debug("iter {}/{}, doc {}/{}".format(i, iterations-1, d, self.num_docs-1))

                self.remove_from_trans_counts(d)
                self.remove_from_topic_counts(d)

                state_log_likes = self.calc_state_topic_log_like_matrix(d)
                state_log_likes += self.calc_state_state_log_like_matrix(d)

                self.doc_states[d] = sample_from_loglikes(state_log_likes)
                self.add_to_trans_counts(d)
                self.add_to_topic_counts(d)

            if i % lag_iters == 0:
                self.save_progress(i, save_dir)

        self.save_progress(i, save_dir)

    # @profile  # used for line-by-line kernprof profiling
    def calc_state_topic_log_like_matrix(self, d):
        # state_topic_counts is (s x t), so each state is a row, each topic a col
        den = self.state_topic_counts + self.alphas  # SxT
        num = den + self.doc_topic_distribs[d]  # SxT
        state_sums = np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den), axis=1)  # Sx1

        num = self.state_topic_totals + self.sum_alpha  # Sx1
        den = num + self.doc_lens[d]  # Sx1
        state_sums += scipy.special.gammaln(num) - scipy.special.gammaln(den)  # Sx1

        return state_sums

    def calc_state_state_log_like_matrix(self, d):
        doc_state = self.doc_states[d]
        doc_prev = self.doc_prevs[d]
        doc_prev_state = self.doc_states[doc_prev] if doc_prev != NULL_DOC else None
        doc_next = self.doc_nexts[d]
        doc_next_state = self.doc_states[doc_next] if doc_next != NULL_DOC else None

        if doc_prev_state is None:  # beginning of resume sequence
            state_liks = self.start_counts + self.pi  # s x 1
            state_liks /= (num_seqs - 1 + self.sum_pi)

            if doc_next_state is not None:  # not a singleton sequence
                state_liks *= self.state_trans[:, doc_next_state] + self.gamma

        else:  # middle of sequence
            if doc_next_state is None:  # end of sequence
                state_liks = self.state_trans[doc_prev_state, :] + self.gamma
            else:
                if doc_prev_state == doc_state:
                    if doc_state == doc_next_state:
                        state_liks = self.state_trans[doc_prev_state, :] + self.gamma
                        state_liks *= self.state_trans[:, doc_next_state] + 1 + self.gamma
                        state_liks /= self.state_trans_tots + 1 + self.sum_gamma
                    else:
                        state_liks = self.state_trans[doc_prev_state, :] + self.gamma
                        state_liks *= self.state_trans[:, doc_next_state] + self.gamma
                        state_liks /= self.state_trans_tots + 1 + self.sum_gamma
                else:  # (doc_prev.state != s)
                    state_liks = self.state_trans[doc_prev_state, :] + self.gamma
                    state_liks *= self.state_trans[:, doc_next_state] + self.gamma
                    state_liks /= self.state_trans_tots + self.sum_gamma

        return np.log(state_liks)

    def init_state_log_like_matrix(self, d):
        doc_prev = self.doc_prevs[d]
        doc_prev_state = self.doc_states[doc_prev] if doc_prev != NULL_DOC else None

        if doc_prev_state is None:  # beginning of resume sequence
            state_liks = self.start_counts + self.pi  # s x 1
            state_liks /= (num_seqs - 1 + self.sum_pi)
        else:  # middle of sequence
            state_liks = self.state_trans[doc_prev_state, :] + self.gamma
        state_logliks = np.log(state_liks)

        state_logliks += self.calc_state_topic_log_like_matrix(d)

        return state_logliks

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
        # Each resume should come in as a list of pairs, where each pair represents a job
        # with (start, end, company, descrip) and (0.04, 0.02, 0.1...) tuples
        resumes = load_json_resumes_lda(infile_name, min_len=min_len, max_entries=max_docs)

        num_docs = sum(map(len, resumes))
        num_topics = len(resumes[0][0][1])  # first resume, first job, length of topic distrib
        logging.info("loading {} resumes, {} jobs, {} topics".format(len(resumes),
                                                                     num_docs,
                                                                     num_topics))

        self.doc_prevs = np.ndarray(num_docs, np.int)
        self.doc_nexts = np.ndarray(num_docs, np.int)
        self.doc_topic_distribs = np.ndarray((num_docs, num_topics))
        self.doc_lens = np.ndarray(num_docs, np.int)

        debug_len_distrib_len = 20
        debug_len_distrib = np.zeros(debug_len_distrib_len, np.int)
        doc_idx = 0
        res_count = 0
        for r, resume in enumerate(resumes):
            resume_len = len(resume)
            debug_len_distrib[min(debug_len_distrib_len-1, resume_len)] += 1
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

        n = float(len(resumes))
        for i in range(debug_len_distrib_len):
            logging.debug("resumes of len {:2d}: {:6d} {:.2f}".format(i, debug_len_distrib[i],
                                                                   debug_len_distrib[i]/n))
        self.num_docs = doc_idx
        self.num_sequences = res_count
        self.doc_states = np.ndarray(self.num_docs, np.int)

    def save_progress(self, i, save_dir):
        ts = str(datetime.datetime.now())

        # model parameters (should not change between iters)
        params = {
            "num_states": self.num_states,
            "pi": self.pi,
            "gamma": self.gamma,
            "alphas": [a for a in self.alphas],
            "num_sequences": self.num_sequences,
        }
        json_str = json.dumps(params)
        append_to_file(os.path.join(save_dir, OUT_PARAMS), [ts, i, json_str])

        # data structures capturing results of latest sampling iteration
        json_str = json.dumps([c for c in self.start_counts])
        append_to_file(os.path.join(save_dir, OUT_START_COUNTS), [ts, i, json_str])

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
        self.pi = params["pi"]
        self.gamma = params["gamma"]
        self.alphas = params["alphas"]
        self.num_sequences = params["num_sequences"]

        ts, iter_starts, json_str = read_last_line(os.path.join(save_dir, OUT_START_COUNTS))
        self.start_counts = np.array(json.loads(json_str), np.int)

        ts, iter_trans, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TRANS))
        self.state_trans = np.array(json.loads(json_str), np.int)

        ts, iter_topics, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TOPICS))
        self.state_topic_counts = np.array(json.loads(json_str), np.double)

        ts, iter_states, json_str = read_last_line(os.path.join(save_dir, OUT_STATES))
        doc_states = json.loads(json_str)
        self.doc_states = np.array(doc_states, np.int)

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
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--num_docs', type=int, default=sys.maxint)

    args = parser.parse_args()

    logging.info("scanning resume file")
    num_seqs, num_docs, num_tops = scan_json_resumes_lda(args.infile, args.min_len, args.num_docs)
    logging.info("{} resumes, {} jobs, {} topics".format(num_seqs, num_docs, num_tops))

    hmm = ResumeHmm(args.num_states, args.pi, args.gamma, num_tops)

    logging.info("loading resumes from file")
    hmm.load_docs_from_resumes(args.infile, min_len=args.min_len, max_docs=args.num_docs)

    logging.info("fitting HMM")
    hmm.fit(args.savedir, args.num_iters, args.lag, erase=args.erase)

    print "yo zzz"



#######################
# TODO:
#
# - seem to get different doc counts from scan and load... need to investigate
#
#
#
#
#





