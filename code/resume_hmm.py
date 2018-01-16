import sys
import math
import argparse
import json
import datetime
import logging
import os
import os.path
import numpy as np
import multiprocessing
import scipy.special

import resume_common
from resume_lda import load_json_resumes_lda


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


OUT_PARAMS = 'params.tsv'
OUT_START_COUNTS = 'starts.tsv'
OUT_STATE_TRANS = 'trans.tsv'
OUT_STATE_TOPICS = 'topics.tsv'
OUT_STATES = 'states.tsv'


def init(p_num_states, p_pi, p_gamma, p_num_topics):
    # We declare these as module-level variables so they will automatically be shared
    # by subprocesses.  A bit ugly way to structure a program, but perhaps a
    # necessary micro-evil.  Possible todo: do away with ResumeHMM class restructure
    # code as a script operating on global variables.  Encapsulation is for wimps!
    global num_states, num_topics, alphas, pi, gamma, sum_alpha, sum_pi, sum_gamma, \
        start_counts, state_trans, state_trans_tots, \
        state_topic_counts, state_topic_totals
    num_states = p_num_states
    num_topics = p_num_topics

    pi = p_pi        # (uniform) prior on start state
    gamma = p_gamma  # (uniform) prior on state-state transitions

    sum_pi = pi*num_states
    sum_gamma = gamma*num_states

    start_counts = np.zeros(num_states, np.int_)
    state_trans = np.zeros((num_states, num_states), np.int_)
    state_trans_tots = np.zeros(num_states, np.int_)

    state_topic_counts = [np.zeros(num_topics)]*num_states
    state_topic_totals = np.zeros(num_states)

    sum_alpha = num_topics
    alphas = np.array([sum_alpha/num_topics]*num_topics)  #zzz Why do it this way? It's 1.0!


def fit(save_dir, iters, iters_lag, erase=False, num_procs=1, chunk_size=None):
    global num_sequences, document_states  #, document_topic_distribs

    num_sequences = len([ d for d in documents if d.doc_prev is None ])
    # document_topic_distribs = [ d.topic_distrib for d in docs ]

    if erase:
        delete_progress(save_dir)
    if os.path.isfile(os.path.join(save_dir, OUT_PARAMS)):
        i = load_progress(save_dir)
        sample_doc_states(save_dir, iters, iters_lag, start_iter=i+1,
                          num_procs=num_procs, chunk_size=chunk_size)
    else:
        init_doc_states(num_procs)
        sample_doc_states(save_dir, iters, iters_lag, num_procs=num_procs, chunk_size=chunk_size)


def init_doc_states(num_procs):
    global documents, document_states

    logging.debug("initializing states for {} documents".format(len(documents)))
    pool = multiprocessing.Pool(processes=num_procs)

    for d in range(len(documents)):

        # print "\n\n\ndoc states: ", [s for s in document_states[:20]]

        if d % 1000 == 0:
            logging.debug("initializing doc {}/{}".format(d, len(documents) - 1))

        args = [(d, s) for s in range(num_states)]
        # logging.debug("{}".format(args))

        # st_log_liks = pool.map(init_state_log_like, args)
        # top_log_liks = pool.map(calc_state_topic_log_like, args)
        #
        # state_log_likes = [s+t for s, t in zip(st_log_liks, top_log_liks)]

        state_log_likes = pool.map(init_state_log_like, args)


        new_state = sample_from_loglikes(state_log_likes)
        # print "doc ", d, " new state: ", new_state
        document_states[d] = new_state

        documents[d].doc_next = 19

        # print "doc states after write doc", d, ": ", [s for s in document_states[:20]]

        init_trans_counts(d)
        add_to_topic_counts(d)

    # for d, doc in enumerate(documents):
    #     state_log_likes = np.zeros(num_states)
    #     for s in range(num_states):
    #         state_log_likes[s] = init_state_log_like(d, s) + \
    #                              calc_state_topic_log_like(s, d)
    #
    #     documents[d].state = sample_from_loglikes(state_log_likes)
    #     init_trans_counts(d)
    #     add_to_topic_counts(d)


# def init_state_log_like(d, s):
def init_state_log_like(params):
    global documents, document_states
    # print "doc states ii: ", [s for s in document_states[:20]]
    # print "doc prevs ii:  ", [doc.doc_prev for doc in documents[:20]]
    # print "doc nexts ii:  ", [doc.doc_next for doc in documents[:20]]


    # print "init", params
    d, s = params
    doc = documents[d]
    # this  is just like calc_state_state_log_like(), except we don't have access to
    # the state of doc_next while initializing, so things are a bit simpler
    if doc.doc_prev is None:  # beginning of resume sequence
        lik = (start_counts[s] + pi) / (num_sequences - 1 + sum_pi)
        # print d, s, "lik beg", lik

    else:

        # print d, s, "trans", state_trans[2][1]

        doc_prev_state = document_states[doc.doc_prev]
        # print d, s, "doc prev", doc.doc_prev, "doc prev state", doc_prev_state

        lik = state_trans[document_states[doc.doc_prev], s] + gamma
        # print d, s, "lik mid", lik

    loglik = math.log(lik)


    loglik += calc_state_topic_log_like(d, s)


    return loglik


def init_trans_counts(d):
    doc = documents[d]
    doc_state = document_states[d]
    if doc.doc_prev is None:  # beginning of resume sequence
        start_counts[doc_state] += 1
    else:  # middle of sequence
        state_trans[document_states[doc.doc_prev], doc_state] += 1

    if doc.doc_next is not None:  # not the end of sequence
        state_trans_tots[doc_state] += 1


def sample_doc_states(save_dir, iterations, lag_iters, start_iter=0, num_procs=1, chunk_size=None):
    timing_iters = 10  # calc a moving average of time per iter over this many
    timing_start = datetime.datetime.now()

    # create a multiprocessing pool that can be reused each iteration
    pool = multiprocessing.Pool(processes=num_procs)

    for i in range(start_iter, iterations):
        logging.debug("iter {}".format(i))
        if i % timing_iters == 0:
            ts_now = datetime.datetime.now()
            logging.debug("current pace {}/iter".format((ts_now - timing_start)//timing_iters))
            timing_start = ts_now

        # for d, doc in enumerate(docs):
        for d in range(len(documents)):
            if d % 1000 == 0:
                logging.debug("iter {}/{}, doc {}/{}".format(i, iterations-1, d, len(documents)-1))

            remove_from_trans_counts(d)
            remove_from_topic_counts(d)

            # if num_procs == 1:
            #     state_log_likes_chunks = [ calc_state_log_like((d, s, s+1))
            #                                for s in range(num_states) ]
            # else:
            #     if chunk_size is None:
            #         chunk_size = num_states/num_procs  # this should give us a floor
            #     state_chunk_starts = range(0, num_states, chunk_size)
            #     state_chunk_ends = [ min(s + chunk_size, num_states) for s in state_chunk_starts ]
            #     args = [(d, start, end) for start, end in zip(state_chunk_starts, state_chunk_ends) ]
            #     # if d % 100 == 0:
            #     #     logging.debug(
            #     #         "args: {}".format(args))
            #     state_log_likes_chunks = pool.map(calc_state_log_like, args)
            #
            # state_log_likes, _ = resume_common.flatten(state_log_likes_chunks)  # todo: use numpy flatten
            if num_procs == 1:
                state_log_likes = [ calc_state_log_like((d, s)) for s in range(num_states) ]
            else:
                args = [ (d, s) for s in range(num_states) ]
                state_log_likes = pool.map(calc_state_log_like, args)

            document_states[d] = sample_from_loglikes(state_log_likes)
            add_to_trans_counts(d)
            add_to_topic_counts(d)

        if i % lag_iters == 0:
            save_progress(i, save_dir)

    pool.terminate()


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

def calc_state_log_like(params):
    d, s = params
    return calc_state_topic_log_like(d, s) + calc_state_state_log_like(d, s)


def calc_state_topic_log_like(d, s):
    ret = 0.0

    den = alphas + state_topic_counts[s]
    num = den + documents[d].topic_distrib
    ret += np.sum(scipy.special.gammaln(num) - scipy.special.gammaln(den))

    ret += math.lgamma(sum_alpha + state_topic_totals[s]) - \
           math.lgamma(sum_alpha + state_topic_totals[s] + documents[d].length)

    return ret


def calc_state_state_log_like(d, s):
    doc = documents[d]
    doc_prev_state = document_states[doc.doc_prev] if doc.doc_prev is not None else None
    doc_next_state = document_states[doc.doc_next] if doc.doc_next is not None else None

    if doc_prev_state is None:  # beginning of resume sequence
        lik = (start_counts[s] + pi) / (num_sequences - 1 + sum_pi)
        if doc_next_state is not None:  # not a singleton sequence
            lik *= state_trans[s, doc_next_state] + gamma

    else:  # middle of sequence
        if doc_next_state is None:  # end of sequence
            lik = (state_trans[doc_prev_state, s] + gamma)
        else:
            if doc_prev_state == s:
                if s == doc_next_state:
                    lik = ((state_trans[doc_prev_state, s] + gamma) *
                           (state_trans[s, doc_next_state] + 1 + gamma) /
                           (state_trans_tots[s] + 1 + sum_gamma))
                else:
                    lik = ((state_trans[doc_prev_state, s] + gamma) *
                           (state_trans[s, doc_next_state] + gamma) /
                           (state_trans_tots[s] + 1 + sum_gamma))

            # if (doc_prev_state == s) and (s == doc_next_state):
            #     lik = ((state_trans[doc_prev_state, s] + gamma) *
            #            (state_trans[s, doc_next_state] + 1 + gamma) /
            #            (state_trans_tots[s] + 1 + sum_gamma))
            #
            # elif doc_prev_state == s:  # and (s != doc_next.state):
            #     lik = ((state_trans[doc_prev_state, s] + gamma) *
            #            (state_trans[s, doc_next_state] + gamma) /
            #            (state_trans_tots[s] + 1 + sum_gamma))
            #
            else:  # (doc_prev.state != s)
                lik = ((state_trans[doc_prev_state, s] + gamma) *
                       (state_trans[s, doc_next_state] + gamma) /
                       (state_trans_tots[s] + sum_gamma))

    # print "lik for state ", s, "(", trace, ")", ": ", lik
    return math.log(lik)


def add_to_trans_counts(d):
    doc = documents[d]
    doc_state = document_states[d]

    if doc.doc_prev is None:  # beginning of resume sequence
        start_counts[doc_state] += 1
        if doc.doc_next is not None:  # not a singleton sequence
            state_trans[doc_state, document_states[doc.doc_next]] += 1
            state_trans_tots[doc_state] += 1
    else:  # middle of sequence
        state_trans[document_states[doc.doc_prev], doc_state] += 1
        if doc.doc_next is not None:  # not the end of sequence
            state_trans[doc_state, document_states[doc.doc_next]] += 1
            state_trans_tots[doc_state] += 1


def remove_from_trans_counts(d):
    doc = documents[d]
    doc_state = document_states[d]
    if doc.doc_prev is None:  # beginning of resume sequence
        start_counts[doc_state] -= 1
        if doc.doc_next is not None:  # not a singleton sequence
            state_trans[doc_state, document_states[doc.doc_next]] -= 1
            state_trans_tots[doc_state] -= 1
    else:  # middle of sequence
        state_trans[document_states[doc.doc_prev], doc_state] -= 1
        if doc.doc_next is not None:  # not end of sequence
            state_trans[doc_state, document_states[doc.doc_next]] -= 1
            state_trans_tots[doc_state] -= 1


def add_to_topic_counts(d):
    doc = documents[d]
    doc_state = document_states[d]
    state_topic_counts[doc_state] += doc.topic_distrib
    state_topic_totals[doc_state] += doc.length


def remove_from_topic_counts(d):
    doc = documents[d]
    doc_state = document_states[d]
    state_topic_counts[doc_state] -= doc.topic_distrib
    state_topic_totals[doc_state] -= doc.length


def save_progress(i, save_dir):
    ts = str(datetime.datetime.now())

    # model parameters (should not change between iters)
    params = {
        "num_states": num_states,
        "pi": pi,
        "gamma": gamma,
        "alphas": alphas.tolist(),
        "num_sequences": num_sequences,
    }
    json_str = json.dumps(params)
    append_to_file(os.path.join(save_dir, OUT_PARAMS), [ts, i, json_str])

    # data structures capturing results of latest sampling iteration
    json_str = json.dumps(start_counts.tolist())
    append_to_file(os.path.join(save_dir, OUT_START_COUNTS), [ts, i, json_str])

    json_str = json.dumps(state_trans.tolist())
    append_to_file(os.path.join(save_dir, OUT_STATE_TRANS), [ts, i, json_str])

    json_str = json.dumps([ s.tolist() for s in state_topic_counts ])
    append_to_file(os.path.join(save_dir, OUT_STATE_TOPICS), [ts, i, json_str])

    json_str = json.dumps([s for s in document_states])
    append_to_file(os.path.join(save_dir, OUT_STATES), [ts, i, json_str])


def load_progress(save_dir):
    global num_states, pi, gamma, alphas, num_sequences, start_counts, state_trans, state_topic_counts, document_states

    ts, iter_params, json_str = read_last_line(os.path.join(save_dir, OUT_PARAMS))
    params = json.loads(json_str)
    num_states = params["num_states"]
    pi = params["pi"]
    gamma = params["gamma"]
    alphas = params["alphas"]
    num_sequences = params["num_sequences"]

    ts, iter_starts, json_str = read_last_line(os.path.join(save_dir, OUT_START_COUNTS))
    start_counts = np.array(json.loads(json_str), np.int_)

    ts, iter_trans, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TRANS))
    state_trans = np.array(json.loads(json_str), np.int_)

    ts, iter_topics, json_str = read_last_line(os.path.join(save_dir, OUT_STATE_TOPICS))
    state_topic_counts = [ np.array(lst) for lst in json.loads(json_str) ]

    ts, iter_states, json_str = read_last_line(os.path.join(save_dir, OUT_STATES))
    doc_states = json.loads(json_str)
    # for doc, state in zip(documents, doc_states):
    #     doc.state = state
    for i, s in enumerate(doc_states):
        document_states[i] = s


    if iter_params == iter_starts == iter_trans == iter_topics == iter_states:
        return iter_params
    else:
        sys.exit("unequal iter counts loaded")


def delete_progress(save_dir):
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


class Document(object):
    def __init__(self, doc_prev, doc_next, topic_distrib):
        self.doc_prev = doc_prev
        self.doc_next = doc_next
        self.topic_distrib = np.array(topic_distrib)
        self.length = sum(topic_distrib)
        # store these in a global array instead to make multiprocessing happy?
        # self.state = None


def get_docs_from_resumes(resume_list, min_len=1):
    global documents, document_states, document_prevs, document_nexts, document_topic_distribs

    debug_len_distrib = np.zeros(20, np.int_)

    docs = []
    doc_idx = 0
    for r, resume in enumerate(resume_list):
        resume_len = len(resume)
        debug_len_distrib[min(19, resume_len)] += 1
        if r % 10000 == 0:
            logging.debug("\t{} {}".format(r, debug_len_distrib))

        if resume_len < min_len:
            continue

        idxs = range(doc_idx, doc_idx + resume_len)
        prevs = [None] + idxs[:-1]
        nexts = idxs[1:] + [None]
        for i, (res_ent, top_dis) in enumerate(resume):
            docs.append(Document(prevs[i], nexts[i], top_dis))

        doc_idx += resume_len

    documents = docs
    document_states = multiprocessing.Array('i', [-1 for d in documents])





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
    # num_states = max([ doc.state for doc in docs ]) + 1
    num_states = max(document_states) + 1
    state_trans_tots = [0] * num_states
    for d, doc in enumerate(docs):
        if doc.doc_next is not None:
            state_trans_tots[document_states[d]] += 1
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
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--proc_chunk', type=int, default=None)

    args = parser.parse_args()

    # get a list of lists of (ResumeEntry, topic_distrib) tuples
    logging.info("loading resumes from file")
    resumes = load_json_resumes_lda(args.infile)
    num_tops = len(resumes[0][0][1])  # distrib for the first job entry in the first resume
    logging.info("extracting documents from resumes")
    get_docs_from_resumes(resumes)

    logging.info("fitting HMM")
    init(args.num_states, args.pi, args.gamma, num_tops)
    fit(args.savedir, args.num_iters, args.lag, erase=args.erase,
        num_procs=args.num_procs, chunk_size=args.proc_chunk)

    print "yo zzz"