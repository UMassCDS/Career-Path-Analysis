import json
import sys
import logging
import datetime
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import resume_common
from resume_import import load_json_resumes


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


# # gets a list of lists of ResumeEntry objs
# def get_descs_flat(resumes):
#     descs = []
#     job_sequence_counts = []
#     total_job_count = 0
#     for t, jobs in enumerate(resumes):
#         job_count = 0
#         # we shouod avoid name-indexing for speed, but this way we're more flexible
#         # for j, (start, end, company_name, desc) in enumerate(jobs):
#         #     descs.append(' '.join(desc))
#         for j, res_ent in enumerate(jobs):
#             descs.append(res_ent.desc)
#             job_count += 1
#             total_job_count += 1
#         if job_count != 0:
#             job_sequence_counts.append(job_count)
#
#     # print out some useful information
#     print '\n'
#     print 'number of job descriptions:', total_job_count
#     print 'number of job description sequences:', len(job_sequence_counts)
#     print '\n'
#     return descs, job_sequence_counts


class ResumeLDA(object):
    # def __init__(self, resume_list, n_topics, normalized=True, n_jobs=4):
    def __init__(self, job_descs_vectored, n_topics, normalized=True, n_jobs=4):
        # jobs, job_sequence_counts = resume_common.flatten(resume_list)
        # job_descs = [j.desc for j in jobs]

        # self.termfreq_vectorizer = CountVectorizer()
        # job_descs_vectored = self.termfreq_vectorizer.fit_transform(job_descs)

        self.lda_model = LatentDirichletAllocation(n_topics=n_topics,
                                              learning_method='batch',
                                              evaluate_every=10,
                                              n_jobs=n_jobs,
                                              verbose=10,
                                              doc_topic_prior=None,
                                              topic_word_prior=None)
        if normalized:
            job_descs_lda = self.lda_model.fit_transform(job_descs_vectored)
        else:
            self.lda_model.fit(job_descs_vectored)
            job_descs_lda, _ = self.lda_model._e_step(job_descs_vectored,
                                                 cal_sstats=False,
                                                 random_init=False)

        print "components_ shape: ", self.lda_model.components_.shape
        print "job_descs_lda shape: ", job_descs_lda.shape

        self.job_descs_lda = job_descs_lda

        # jobs_lda = zip(jobs, job_descs_lda)
        # self.jobs_lda_seq = resume_common.unflatten(jobs_lda, job_sequence_counts)


#
#
# def transform_descs_lda(resume_list, n_topics=200, n_jobs=4, normalized=True):
#     # job_descs, job_sequence_counts = get_descs_flat(resume_list)
#     jobs, job_sequence_counts = resume_common.flatten(resume_list)
#     job_descs = [ j.desc for j in jobs ]
#
#     termfreq_vectorizer = CountVectorizer()
#     job_descs_vectored = termfreq_vectorizer.fit_transform(job_descs)
#
#     print "job_descs_vectored shape: ", job_descs_vectored.shape
#     print "doc 0", job_descs[0], job_descs_vectored[0, :], "\n"
#     print "doc 1", job_descs[1], job_descs_vectored[1, :], "\n"
#     print "doc 2", job_descs[2], job_descs_vectored[2, :], "\n"
#
#
#     lda_model = LatentDirichletAllocation(n_topics=n_topics,
#                                           learning_method='batch',
#                                           evaluate_every=10,
#                                           n_jobs=n_jobs,
#                                           verbose=10,
#                                           doc_topic_prior=None,
#                                           topic_word_prior=None)
#     if normalized:
#         job_descs_lda = lda_model.fit_transform(job_descs_vectored)
#     else:
#         lda_model.fit(job_descs_vectored)
#         job_descs_lda, _ = lda_model._e_step(job_descs_vectored,
#                                       cal_sstats=False,
#                                       random_init=False)
#
#
#
#
#
#     # for d, (desc, lda) in enumerate(zip(job_descs, job_descs_lda)):
#     #     print "lda_topic_distrib", d, sum(lda), len(desc.split()), ": ", lda
#
#     print "components_ shape: ", lda_model.components_.shape
#
#     print "job_descs_lda shape: ", job_descs_lda.shape
#     # for lda_distrib in job_descs_lda[:10]:
#     #     print lda_distrib, ", sum=", sum(lda_distrib), [d*len(desc.split()) for d in lda_distrib], len(desc.split()), desc, "\n"
#
#     jobs_lda = zip(jobs, job_descs_lda)
#     jobs_lda_seq = resume_common.unflatten(jobs_lda, job_sequence_counts)
#
#     # job_descs_lda_seq = unflatten(job_descs_lda, job_sequence_counts)
#     # if normalized:
#     # else:
#     #     job_descs_lda_counts = [ [t*len(desc.split()) for t in lda] for desc, lda in zip(job_descs, job_descs_lda)]
#     # print "job_descs_lda_counts shape: ", job_descs_lda.shape
#     # print job_descs_lda_counts[:3]
#     #
#     #     for desc, lda_distrib in zip(job_descs, job_descs_lda)[:10]:
#     #         print lda_distrib, [d*len(desc.split()) for d in lda_distrib], len(desc.split()), desc, "\n"
#     #
#     #     job_descs_lda_seq = unflatten(job_descs_lda_counts, job_sequence_counts)
#     # return job_descs_lda_seq
#
#     dump_topic_word_distribs(lda_model, termfreq_vectorizer, "topic_word_distribs.json", threshold=0.1)
#
#     return jobs_lda_seq


# def dump_topic_word_distribs(resume_lda, outfile_name, threshold=1.1):
#     lda_model = resume_lda.lda_model
#     word_vectorizer = resume_lda.termfreq_vectorizer
def dump_topic_word_distribs(lda_model, word_vectorizer, outfile_name, threshold=1.1):

    topic_distribs = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    num_topics, num_words = topic_distribs.shape

    word_names = word_vectorizer.get_feature_names()

    with open(outfile_name, 'w') as outfile:
        for topic_id in range(num_topics):
            topic_distrib = topic_distribs[topic_id, :]
            word_freqs = [ (f, word_names[w]) for w, f in enumerate(topic_distrib) ]
            word_freqs.sort(reverse=True)

            topic_words = []
            mass = 0.0
            for freq, word in word_freqs:
                topic_words.append((word, freq))
                mass += freq
                if mass > threshold:
                    break
            json_str = json.dumps([topic_id, topic_words])
            outfile.write(json_str + "\n")

            try:
                topic_words_str = ", ".join(["{} {:0.4f}".format(w, f) for w, f in topic_words])
                print "topic {}:\t{}\n".format(topic_id, topic_words_str)
            except UnicodeEncodeError as e:
                print "topic {}: ERR".format(topic_id)


def read_topic_word_distribs(infile_name, threshold=1.1):
    topic_word_distribs_unordered = []
    max_id = -1
    with open(infile_name, 'r') as infile:
        for line in infile:
            topic_id, word_freqs_raw = json.loads(line.rstrip("\n"))
            word_freqs = sorted([(w, f) for w, f in word_freqs_raw],
                                reverse=True, key=lambda x: x[1])
            pos = 0
            mass = 0.0
            for word, freq in word_freqs:
                pos += 1
                mass += freq
                if mass > threshold:
                    break
            if topic_id > max_id:
                max_id = topic_id
            topic_word_distribs_unordered.append((topic_id, word_freqs[:pos]))
    topic_word_distribs = [None]*(max_id+1)
    for topic_id, word_freqs in topic_word_distribs_unordered:
        topic_word_distribs[topic_id] = word_freqs
    return topic_word_distribs


# Even though we only use dump() here, define them together so they stay in sync
def dump_json_resumes_lda(resumes, outfile_name):
    with open(outfile_name, 'w') as outfile:
        # out = [ [ [resume_common.tuplify(e), lda.tolist()] for e, lda in resume ] for resume in resumes ]
        # json.dump(out, outfile)
        for resume in resumes:
            json_line = json.dumps([ [resume_common.tuplify(e), lda.tolist()] for e, lda in resume ])
            outfile.write(json_line + "\n")


def load_json_resumes_lda(infile_name, min_len=1, max_entries=sys.maxint):
    resume_tups = []
    entry_count = 0
    with open(infile_name, 'r') as infile:
        # resume_tups = json.load(infile)
        for json_line in infile:
            res = json.loads(json_line.rstrip("\n"))
            if len(res) < min_len:
                continue
            resume_tups.append(res)
            entry_count += len(res)
            if entry_count > max_entries:
                break
    return [ [ (resume_common.detuplify(e), lda) for e, lda in entry ] for entry in resume_tups ]


def scan_json_resumes_lda(infile_name, min_len=1, max_entries=sys.maxint):
    resume_count = 0
    entry_count = 0
    topic_count = None
    with open(infile_name, 'r') as infile:
        for json_line in infile:
            resume = json.loads(json_line.rstrip("\n"))
            resume_len = len(resume)
            if resume_len < min_len:
                continue
            entry_count += resume_len
            resume_count += 1
            topic_count = len(resume[0][1])
            if entry_count > max_entries:
                break
    return resume_count, entry_count, topic_count


def ts(spc=1):
    diff = datetime.datetime.now() - ts.t0
    return " "*spc + str(diff - datetime.timedelta(microseconds=diff.microseconds)) + \
        "{:0.2f}".format(diff.microseconds/1000000.0)[1:] + " "*spc
ts.t0 = datetime.datetime.now()


###################################
if __name__ == '__main__':

    USAGE = " usage: " + sys.argv[0] + " resumes.json resumes_lda.json topics_lda.json num_topics num_jobs"
    if len(sys.argv) < 5:
        sys.exit(USAGE)
    infile_name = sys.argv[1]
    outfile_name = sys.argv[2]
    topicfile_name = sys.argv[3]
    num_topics = int(sys.argv[4])
    num_jobs = int(sys.argv[5])

    # with open(infile_name, 'rb') as infile:
    #     resumes_raw = p.load(infile)
    # resume_list = []
    # for res in resumes_raw:
    #     resume = []
    #     for entry in res:
    #         resume.append(resume_common.ResumeEntry(*entry))
    #     resume_list.append(resume)
    logging.info(ts() + "loading raw resumes")
    resume_list = load_json_resumes(infile_name)  # list of lists of ResumeEntry

    # jobs_lda_sequenced = transform_descs_lda(resume_list,  # list of lists of topic distribs
    #                                          n_topics=num_topics,
    #                                          n_jobs=num_jobs,
    #                                          normalized=False)

    # we do this outside of the class so we can be aggressive about garbage collecting before
    # we start multiprocessing within scikit's LDA
    logging.info(ts() + "flattening resumes")
    jobs, job_sequence_counts = resume_common.flatten(resume_list)
    del resume_list

    logging.info(ts() + "vectoring descriptions")
    termfreq_vectorizer = CountVectorizer()
    job_descs_vectored = termfreq_vectorizer.fit_transform([j.desc for j in jobs])
    del jobs
    del job_sequence_counts

    logging.info(ts() + "learning lda model")
    lda = ResumeLDA(job_descs_vectored, num_topics, normalized=False, n_jobs=num_jobs)

    logging.info(ts() + "reloading raw resumes")
    resume_list = load_json_resumes(infile_name)
    jobs, job_sequence_counts = resume_common.flatten(resume_list)

    jobs_lda = zip(jobs, lda.job_descs_lda)
    jobs_lda_seq = resume_common.unflatten(jobs_lda, job_sequence_counts)

    logging.info(ts() + "dumping output")
    dump_json_resumes_lda(jobs_lda_seq, outfile_name)
    dump_topic_word_distribs(lda.lda_model, termfreq_vectorizer, topicfile_name, threshold=0.5)

# with open(outfile_name, 'w') as outfile:
    #     json.dump(job_descs_lda_sequenced, outfile)



