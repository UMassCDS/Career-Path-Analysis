import sys
import json
import numpy as np
from resume_lda import read_topic_word_distribs


# print the average topic mix for each state
def print_state_descs(state_topic_file_name, topic_word_file_name):

    topic_words = read_topic_word_distribs(topic_word_file_name)

    with open(state_topic_file_name, 'r') as state_topic_file:
        iter0 = json.loads(state_topic_file.readline().rstrip("\n"))  # SxT 2d arr
        state_topic_sums = np.array(iter0, np.double)
        iter_count = 1

        for line in state_topic_file:
            state_topic_sums += np.array(json.loads(line.rstrip("\n")), np.double)
            iter_count += 1
    state_topic_avg = state_topic_sums / iter_count

    for s in range(state_topic_avg.shape[0]):
        print "state {} topics:".format(s)
        perc_top_tups = sorted([ (p, i) for i, p in enumerate(state_topic_avg[s].tolist()) ],
                               reverse=True)
        for perc, top in perc_top_tups[:10]:
            print "{:0.2f}\t{}".format(perc, "  ".join(topic_words[top]))
        print "\n"



########################################################
if __name__ == '__main__':

    s_t_file_name = sys.argv[1]
    t_w_file_name = sys.argv[2]


    print_state_descs(s_t_file_name, t_w_file_name)


