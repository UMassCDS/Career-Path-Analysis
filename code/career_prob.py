import logging
import argparse
import numpy as np
import scipy.stats
import resume_lda
import resume_import_db as impdb
import resume_lda_db as ldadb
import resume_results


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


TRANS_FILE = '/data/output/hmm_topics200_states500_iters10k_docs30k/trans.tsv'
STATE_FILE = '/data/output/hmm_topics200_states500_iters10k_docs30k/states.tsv'
LDA_FILE = '/data/output/output_all_200/resumes_all_lda200.json'

TRANS_PROB_TABLE = "job_transition_probs"

GAMMA = 1.0
BURN = 1000
LAG = 10
MIN_RESUME_LEN = 3
MAX_ENTRIES = 30000


# MAP would be: P(next|state) = P(state|next)P(next) / P(state)
# MLE is: p_ij = n_ij / SUM[ n_ik ] for k=0 to num_states
def trans_prob(state, state_next, state_state_trans):
    num_states = state_state_trans.shape[0]
    n_ij = state_state_trans[state, state_next] + GAMMA
    sum_n_ik = sum([state_state_trans[state, k] + GAMMA for k in range(num_states)])
    return float(n_ij) / sum_n_ik


def create_trans_prob_table(conn, overwrite=False):
    table_name = TRANS_PROB_TABLE
    curs = conn.cursor()
    if overwrite:
        curs.execute("DROP TABLE IF EXISTS " + table_name)
    sql = "CREATE TABLE " + table_name + " (job_id1 INTEGER, job_id2 INTEGER, job_state1 INTEGER, job_state2 INTEGER, trans_prob FLOAT)"
    curs.execute(sql)


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

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost')
parser.add_argument('--user', default=None)
parser.add_argument('--db', default=None)
args = parser.parse_args()
logging.info("connecting to db")
conn = impdb.get_connection(args.host, args.db, args.user)

res_dbs = ldadb.get_resumes_db(conn)
job_id_hash = ldadb.get_job_id_hash(res_dbs)

create_trans_prob_table(conn, overwrite=True)

res_tups = []
job_idx = 0

curs = conn.cursor()
hits = 0
misses = 0
for r, resume in enumerate(resumes):
    if r % 1000 == 0:
        logging.debug("\t{}/{} ({} hits, {} misses)".format(r, len(resumes)-1, hits, misses))
        conn.commit()

    res_key = ldadb.make_resume_date_key_lda(resume)
    if res_key in job_id_hash:
        hits += 1

        resume_id, job_ids = job_id_hash[res_key]
        job_states = []
        # job_companies = []
        # trans_probs = []

        for i, job in enumerate(resume):
            # logging.debug("{}".format(resume))
            # res_ent, top_dis = job
            # start, end, company, desc = res_ent
            # logging.debug("res {} job {}: {}".format(r, job_idx, company))
            job_state = doc_states[job_idx]
            job_states.append(job_state)
            # job_companies.append(company)
            job_idx += 1

        # trans_product = 1.0
        for i in range(1, len(job_states)):
            prob = trans_prob(job_states[i-1], job_states[i], state_state_trans)
            # trans_product *= prob
            tup = (job_ids[i-1], job_ids[i], job_states[i-1], job_states[i], prob)
            curs.execute("INSERT INTO " + TRANS_PROB_TABLE + " VALUES(%s, %s, %s, %s, %s)", tup)

    else:
        misses += 1


        # logging.debug("{}\t{}\t{}\n".format(" => ".join(job_companies), len(job_states), trans_product))
        # res_tups.append((len(job_states), trans_product, job_companies))

# res_tups.sort()
# for num_jobs, prob, companies in res_tups:
#     logging.debug("{}\t{}\t{}\n".format(num_jobs, trans_product, " => ".join(companies)))

