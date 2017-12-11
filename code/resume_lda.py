import sys
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import resume_common


# gets a list of lists of ResumeEntry objs
def get_descs_flat(resumes):
    descs = []
    job_sequence_counts = []
    total_job_count = 0
    for t, jobs in enumerate(resumes):
        job_count = 0
        # we shouod avoid name-indexing for speed, but this way we're more flexible
        # for j, (start, end, company_name, desc) in enumerate(jobs):
        #     descs.append(' '.join(desc))
        for j, res_ent in enumerate(jobs):
            descs.append(res_ent.desc)
            job_count += 1
            total_job_count += 1
        if job_count != 0:
            job_sequence_counts.append(job_count)

    # print out some useful information
    print '\n'
    print 'number of job descriptions:', total_job_count
    print 'number of job description sequences:', len(job_sequence_counts)
    print '\n'
    return descs, job_sequence_counts


def unflatten(flat_list, sublist_lens):
    rets = []
    idx = 0
    for sublist_len in sublist_lens:
        rets.append(flat_list[idx:(idx+sublist_len)].tolist())
        idx += sublist_len
    return rets


def transform_descs_lda(resume_list, n_topics=200, n_jobs=4):
    job_descs, job_sequence_counts = get_descs_flat(resume_list)

    termfreq_vectorizer = CountVectorizer()
    job_descs_vectored = termfreq_vectorizer.fit_transform(job_descs)

    lda_model = LatentDirichletAllocation(n_topics=n_topics,
                                          learning_method='batch',
                                          evaluate_every=10,
                                          n_jobs=n_jobs,
                                          verbose=10)
    job_descs_lda = lda_model.fit_transform(job_descs_vectored)
    job_descs_lda_sequenced = unflatten(job_descs_lda, job_sequence_counts)
    return job_descs_lda_sequenced


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
    resume_list = resume_common.load_json_file(infile_name)

    job_descs_lda_sequenced = transform_descs_lda(resume_list, n_topics=20, n_jobs=4)

    with open(outfile_name, 'w') as outfile:
        json.dump(job_descs_lda_sequenced, outfile)



