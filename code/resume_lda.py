import json
import sys

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import resume_common
from resume_import import load_json_resumes


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


def transform_descs_lda(resume_list, n_topics=200, n_jobs=4, normalized=True):
    # job_descs, job_sequence_counts = get_descs_flat(resume_list)
    jobs, job_sequence_counts = resume_common.flatten(resume_list)
    job_descs = [ j.desc for j in jobs ]

    termfreq_vectorizer = CountVectorizer()
    job_descs_vectored = termfreq_vectorizer.fit_transform(job_descs)

    print "job_descs_vectored shape: ", job_descs_vectored.shape
    print job_descs_vectored[:3]


    lda_model = LatentDirichletAllocation(n_topics=n_topics,
                                          learning_method='batch',
                                          evaluate_every=10,
                                          n_jobs=n_jobs,
                                          verbose=10,
                                          doc_topic_prior=None,
                                          topic_word_prior=None)
    if normalized:
        job_descs_lda = lda_model.fit_transform(job_descs_vectored)
    else:
        lda_model.fit(job_descs_vectored)
        job_descs_lda, _ = lda_model._e_step(job_descs_vectored,
                                      cal_sstats=False,
                                      random_init=False)

    for d, (desc, lda) in enumerate(zip(job_descs, job_descs_lda)):
        print "lda_topic_distrib", d, sum(lda), len(desc.split()), ": ", lda

    print "job_descs_lda shape: ", job_descs_lda.shape
    # for lda_distrib in job_descs_lda[:10]:
    #     print lda_distrib, ", sum=", sum(lda_distrib), [d*len(desc.split()) for d in lda_distrib], len(desc.split()), desc, "\n"

    jobs_lda = zip(jobs, job_descs_lda)
    jobs_lda_seq = resume_common.unflatten(jobs_lda, job_sequence_counts)

    # job_descs_lda_seq = unflatten(job_descs_lda, job_sequence_counts)
    # if normalized:
    # else:
    #     job_descs_lda_counts = [ [t*len(desc.split()) for t in lda] for desc, lda in zip(job_descs, job_descs_lda)]
    # print "job_descs_lda_counts shape: ", job_descs_lda.shape
    # print job_descs_lda_counts[:3]
    #
    #     for desc, lda_distrib in zip(job_descs, job_descs_lda)[:10]:
    #         print lda_distrib, [d*len(desc.split()) for d in lda_distrib], len(desc.split()), desc, "\n"
    #
    #     job_descs_lda_seq = unflatten(job_descs_lda_counts, job_sequence_counts)
    # return job_descs_lda_seq

    return jobs_lda_seq


# Even though we only use dump() here, define them together so they stay in sync
def dump_json_resumes_lda(resumes, outfile_name):
    with open(outfile_name, 'w') as outfile:
        # out = [ [ [resume_common.tuplify(e), lda.tolist()] for e, lda in resume ] for resume in resumes ]
        # json.dump(out, outfile)
        for resume in resumes:
            json_line = json.dumps([ [resume_common.tuplify(e), lda.tolist()] for e, lda in resume ])
            outfile.write(json_line + "\n")


def load_json_resumes_lda(infile_name, max_resumes=sys.maxint):
    resume_tups = []
    resume_count = 0
    with open(infile_name, 'r') as infile:
        # resume_tups = json.load(infile)
        for json_line in infile:
            res = json.loads(json_line.rstrip("\n"))
            resume_tups.append(res)
            resume_count += len(res)
            if resume_count > max_resumes:
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


###################################
if __name__ == '__main__':

    USAGE = " usage: " + sys.argv[0] + " resumes.json jobs_lda.json num_topics num_jobs"
    if len(sys.argv) < 5:
        sys.exit(USAGE)
    infile_name = sys.argv[1]
    outfile_name = sys.argv[2]
    num_topics = int(sys.argv[3])
    num_jobs = int(sys.argv[4])

    # with open(infile_name, 'rb') as infile:
    #     resumes_raw = p.load(infile)
    # resume_list = []
    # for res in resumes_raw:
    #     resume = []
    #     for entry in res:
    #         resume.append(resume_common.ResumeEntry(*entry))
    #     resume_list.append(resume)
    resume_list = load_json_resumes(infile_name)  # list of lists of ResumeEntry

    jobs_lda_sequenced = transform_descs_lda(resume_list,  # list of lists of topic distribs
                                             n_topics=num_topics,
                                             n_jobs=num_jobs,
                                             normalized=False)

    dump_json_resumes_lda(jobs_lda_sequenced, outfile_name)

    # with open(outfile_name, 'w') as outfile:
    #     json.dump(job_descs_lda_sequenced, outfile)



