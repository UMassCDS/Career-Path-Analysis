import sys
import json
import numpy as np
# import seaborn as sns
import logging
import os.path
# import matplotlib.pyplot as plt
from resume_lda import read_topic_word_distribs


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
# sns.set(color_codes=True)


def inspect_output(infile_name, shape_iter_idx=0, delim="\t"):
    with open(infile_name, 'r') as infile:
        # ts, i, json_str = infile.readline().rstrip("\n").split(delim)
        # output_array = np.array(json.loads(json_str), np.double)
        # shape = output_array.shape
        shape = None
        count = 0
        for idx, line in enumerate(infile):
            count += 1
            if idx == shape_iter_idx:
                ts, i, json_str = line.rstrip("\n").split(delim)
                output_array = np.array(json.loads(json_str), np.double)
                shape = output_array.shape
        logging.debug("output shape: {}, {} iters".format(shape, count))


def get_mean_vals(infile_name, burn=0, lag=0, delim="\t"):
    with open(infile_name, 'r') as infile:
        ts, i, json_str = infile.readline().rstrip("\n").split(delim)
        sum_array = np.array(json.loads(json_str), np.double)
        count = 1

        for line in infile:
            if count % 100 == 0:
                print "\t", count
            ts, i, json_str = line.rstrip("\n").split(delim)
            sum_array += json.loads(json_str)
            count += 1
    return sum_array / count


def get_output_vals(infile_name, burn=0, lag=0, dtype=np.double, delim="\t"):
    with open(infile_name, 'r') as infile:
        iters = []

        for iter, line in enumerate(infile):
            if iter < burn:
                continue
            if iter % lag == 0:
                ts, i, json_str = line.rstrip("\n").split(delim)
                iters.append(json.loads(json_str))

    logging.debug("got {} iters of output".format(len(iters)))
    for i in range(5):
        logging.debug("iter {}: {}".format(i, iters[i]))
        
    return np.array(iters, dtype)


# print the average topic mix for each state
def print_state_descs(state_topic_file_name, topic_word_file_name):

    topic_words = read_topic_word_distribs(topic_word_file_name)

    # with open(state_topic_file_name, 'r') as state_topic_file:
    #     ts, i, line0 = state_topic_file.readline().rstrip("\n").split("\t")
    #     # print "line0: ", line0[:200]
    #     iter0 = json.loads(line0)
    #     # print "iter0: ", iter0[:10]
    #     state_topic_sums = np.array(iter0, np.double)
    #     iter_count = 1
    #
    #     for line_num, line in enumerate(state_topic_file):
    #         if line_num % 10 == 0:
    #             print "\t", line_num
    #         ts, i, json_str = line.rstrip("\n").split("\t")
    #         state_topic_sums += np.array(json.loads(json_str), np.double)
    #         iter_count += 1
    # state_topic_avg = state_topic_sums / iter_count
    state_topic_avg = get_mean_vals(state_topic_file_name)

    for s in range(state_topic_avg.shape[0]):
        print "state {} topics:".format(s)
        count_topic_tups = sorted([ (p, i) for i, p in enumerate(state_topic_avg[s].tolist()) ],
                                  reverse=True)
        norm = sum(t[0] for t in count_topic_tups)
        # print count_topic_tups[:20]
        for count, topic in count_topic_tups[:10]:
            # print "count: ", count, "topic: ", topic, "words: ", topic_words[topic]
            perc = count/norm
            words = [ "{}({:.2f})".format(w, f) for w, f in topic_words[topic][:6] ]
            print "{:0.2f}\t{}".format(perc, ",  ".join(words))
        print "\n"

    return state_topic_avg


def get_state_topics(state_topic_file_name, num=10):
    state_topic_avg = get_mean_vals(state_topic_file_name)
    top_topics = []
    for s in range(state_topic_avg.shape[0]):
        count_topic_tups = sorted([(c, i) for i, c in enumerate(state_topic_avg[s].tolist())],
                                  reverse=True)
        top_topics.append(sorted([ i for c, i in count_topic_tups[:num] ]))
    return top_topics


def get_state_descs(state_topic_file_name, topic_word_file_name, num_words):
    topic_words = read_topic_word_distribs(topic_word_file_name)
    state_topic_avg = get_mean_vals(state_topic_file_name)

    state_words = []
    for s in range(state_topic_avg.shape[0]):
        count_topic_tups = sorted([ (c, i) for i, c in enumerate(state_topic_avg[s].tolist()) ],
                                  reverse=True)
        norm = sum(t[0] for t in count_topic_tups)
        word__score = {}
        for count, topic in count_topic_tups:
            perc = count/norm

            for word, freq in topic_words[topic][:500]:
                word__score[word] = word__score.get(word, 0.0) + freq*perc

        words = sorted(word__score.items(), key=lambda x: x[1], reverse=True)
        state_words.append(words[:num_words])
    return state_words


def print_top_trans(state_state_file_name, state_descs, state_topics):
    state_state_avg = get_mean_vals(state_state_file_name)

    # for s1 in range(state_state_avg.shape[0]):
    #     s2_freq_tups = sorted([(s, f) for s, f in enumerate(state_state_avg[s1])],
    #                           key=lambda x: x[1], reverse=True)
    #
    #     print "state", s1, state_descs[s1], "\n"
    #     for s2, f in s2_freq_tups[:5]:
    #         print "\t{:.2f} {}".format(f, s2), state_descs[s2], "\n"
    #
    #     print "\n\n"

    # sns.distplot(state_state_avg.flatten())
    # plt.show()

    trans_tups = []
    for s1 in range(state_state_avg.shape[0]):
        trans_tups.extend([ (freq, s1, s2) for s2, freq in enumerate(state_state_avg[s1]) ])
    print "got {} trans tups".format(len(trans_tups))
    print_count = 0
    for freq, s1, s2 in sorted(trans_tups, reverse=True):
        if s1 != s2:
            print "({}) {} {} => \t{} {}\t{}".format(print_count, s1, state_topics[s1],
                                                     s2, state_topics[s2], freq)

            for desc_tup1, desc_tup2 in zip(state_descs[s1], state_descs[s2]):
                print "\t{} {:30} {} {:30}".format(desc_tup1[1], desc_tup1[0],
                                                   desc_tup2[1], desc_tup2[0])

            print "\n"
            print_count += 1
        if print_count > 100:
            break

    # how many states account for x% of transitions?
    mass_threshold = 0.5
    state_count_tups = []  # (state, # of states that account for x% of transitions)

    for s1 in range(state_state_avg.shape[0]):
        trans_counts = state_state_avg[s1].tolist()
        del trans_counts[s1]  # ignore self trans
        mass_total = sum(trans_counts)
        mass_curr = 0.0
        mass_count = 0
        for mass in sorted(trans_counts, reverse=True):
            mass_curr += mass
            mass_count += 1
            if mass_curr / mass_total > mass_threshold:
                break
        state_count_tups.append((s1, mass_count))

    # most mobile states:
    print "most mobile:"
    for state, count in sorted(state_count_tups, key=lambda x: x[1], reverse=True)[:20]:
        print "state {}\t{}\t{} states make {} mass\n".format(state, count, state_descs[state],
                                                            mass_threshold)

    # least mobile states:
    print "\nleast mobile:"
    for state, count in sorted(state_count_tups, key=lambda x: x[1])[:20]:
        print "state {}\t{}\t{} states make {} mass\n".format(state, count, state_descs[state],
                                                                mass_threshold)








########################################################
if __name__ == '__main__':

    # s_t_file_name = sys.argv[1]
    # t_w_file_name = sys.argv[2]
    # s_s_file_name = sys.argv[3]
    #
    # # print_state_descs(s_t_file_name, t_w_file_name)
    # state_descs = get_state_descs(s_t_file_name, t_w_file_name, 20)
    #
    # state_topics = get_state_topics(s_t_file_name, num=10)
    #
    #
    # # print state_descs
    #
    # state_descs_str = [ [(w, float("{:.4f}".format(f))) for w, f in sd] for sd in state_descs]
    #
    # # for i, words in enumerate(state_descs):
    # #     print i, ["{}({:.4f})".format(w, f) for w, f in words], "\n"
    #
    # print_top_trans(s_s_file_name, state_descs_str, state_topics)

    output_dir = sys.argv[1]
    burn_iters = int(sys.argv[2])
    lag_iters = int(sys.argv[3])

    # generate modal mean state trans counts
    state_state_trans = np.mean(get_output_vals(os.path.join(output_dir, 'trans.tsv'),
                                                burn_iters, lag_iters, np.double), axis=0)





