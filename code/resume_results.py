import sys
import json
import numpy as np
from resume_lda import read_topic_word_distribs


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







########################################################
if __name__ == '__main__':

    s_t_file_name = sys.argv[1]
    t_w_file_name = sys.argv[2]


    # print_state_descs(s_t_file_name, t_w_file_name)
    state_descs = get_state_descs(s_t_file_name, t_w_file_name, 20)
    for i, words in enumerate(state_descs):
        print i, ["{}({:.4f})".format(w, f) for w, f in words], "\n"



