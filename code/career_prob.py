import logging
import numpy as np
import scipy.stats
import resume_lda
import resume_results


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


TRANS_FILE = '/data/output/hmm_topics200_states500_iters10k_docs30k/trans.tsv'
STATE_FILE = '/data/output/hmm_topics200_states500_iters10k_docs30k/states.tsv'
LDA_FILE = '/data/output/output_all_200/resumes_all_lda200.json'

BURN = 1000
LAG = 10
MIN_RESUME_LEN = 3
MAX_ENTRIES = 30000


# MAP would be: P(next|state) = P(state|next)P(next) / P(state)
# MLE is: p_ij = n_ij / SUM[ n_ik ] for k=0 to num_states
def trans_prob(state, state_next, state_state_trans):
    num_states = state_state_trans.shape[0]
    n_ij = state_state_trans[state, state_next]
    sum_n_ik = sum([state_state_trans[state, k] for k in range(num_states)])
    return float(n_ij) / sum_n_ik


resume_results.inspect_output(STATE_FILE, shape_iter_idx=15000)

logging.debug("aggreg doc states")
doc_states, _ = scipy.stats.mode(resume_results.get_output_vals(STATE_FILE, BURN, LAG, np.integer),
                              axis=0)
doc_states = doc_states[0]
logging.debug("doc states {}".format(doc_states.shape))

logging.debug("aggreg state trans")
state_state_trans = np.mean(resume_results.get_output_vals(TRANS_FILE, BURN, LAG, np.double),
                            axis=0)
logging.debug("state trans {}".format(state_state_trans.shape))

logging.debug("loading resumes")
resumes = resume_lda.load_json_resumes_lda(LDA_FILE,
                                           min_len=MIN_RESUME_LEN, max_entries=MAX_ENTRIES)

res_tups = []
job_idx = 0

for r, resume in enumerate(resumes):
    job_states = []
    job_companies = []
    for i, job in enumerate(resume):
        # logging.debug("{}".format(resume))
        res_ent, top_dis = job
        start, end, company, desc = res_ent
        logging.debug("res {} job {}: {}".format(r, job_idx, company))
        job_state = doc_states[job_idx]
        job_states.append(job_state)
        job_companies.append(company)
        job_idx += 1

    trans_product = 1.0
    for i in range(1, len(job_states)):
        trans_product *= trans_prob(job_states[i-1], job_states[i], state_state_trans)
    logging.debug("{}\t{}\t{}\n".format(" => ".join(job_companies), len(job_states), trans_product))

    res_tups.append((len(job_states), trans_product, job_companies))

res_tups.sort()
for num_jobs, prob, companies in res_tups:
    logging.debug("{}\t{}\t{}\n".format(num_jobs, trans_product, " => ".join(companies)))

