import argparse
import logging
import json
import resume_common
import resume_lda
import resume_import_db as impdb


LDA_FILE = '/data/output/output_all_200/resumes_all_lda200.json'


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def get_resumes_lda(infile_name):
    with open(infile_name, 'r') as infile:
        for json_line in infile:
            res = json.loads(json_line.rstrip("\n"))
            yield [ (resume_common.detuplify(e), lda) for e, lda in res ]


def get_resumes_db(conn):
    curs = conn.cursor()
    sql = "SELECT job_id, resume_id, start_dt, end_dt, description FROM jobs ORDER BY job_id"
    logging.debug(sql)
    curs.execute(sql)

    rec = curs.fetchone()
    resume_curr = [rec]
    resume_id_prev = rec[1]
    for rec in curs:
        resume_id_curr = rec[1]
        if resume_id_curr == resume_id_prev:
            resume_curr.append(rec)
        else:
            resume_ret = resume_curr
            resume_curr = [rec]
            resume_id_prev = resume_id_curr
            yield resume_ret


def make_resume_date_key_lda(resume):
    start_end_pairs = [ (job[0][0], job[0][1]) for job in resume ]
    return tuple(sorted(start_end_pairs))


def make_resume_date_key_db(resume):
    start_end_pairs = [(job[2], job[3]) for job in resume]
    return tuple(sorted(start_end_pairs))


def dump_res_lda(res):
    for j, job in enumerate(res):
        logging.debug("res lda ({}/{}): {}".format(j, len(res)-1, job)[:150])


def dump_res_db(res):
    for j, job in enumerate(res):
        logging.debug("res db  ({}/{}): {}".format(j, len(res)-1, job)[:150])


def get_job_id_hash(resumes):
    date_key__id = {}
    collisions = 0
    logging.debug("creating job id hash from db")
    for r, resume in enumerate(resumes):
        if r % 50000 == 0:
            logging.debug("\t{}".format(r))

        job_ids = [ job[0] for job in resume ]
        key = make_resume_date_key_db(resume)
        if key in date_key__id:
            logging.debug("COLLISION {} (new):   {}".format(collisions, resume))
            logging.debug("COLLISION {} (exist): {}\n".format(collisions, date_key__id[key]))
            collisions += 1
        else:
            date_key__id[key] = job_ids
    return date_key__id


def marry_lda_db(conn):
    # logging.info("loading lda resumes")
    # res_ldas = resume_lda.load_json_resumes_lda(LDA_FILE, min_len=0)
    res_dbs = get_resumes_db(conn)

    # for res_lda in res_ldas:
    for r, res_lda in enumerate(get_resumes_lda(LDA_FILE)):

        if r % 1000 == 0:
            logging.debug("\t{}".format(r))

        res_db = res_dbs.next()

        if len(res_lda) == len(res_db):
            for lda_job, db_job in zip(res_lda, res_db):
                lda_res_ent, lda_output = lda_job
                lda_start, lda_end, lda_company, lda_desc = lda_res_ent
                db_job_id, db_resume_id, db_start, db_end, db_desc = db_job

                if (lda_start == db_start) and (lda_end == db_end):
                    print json.dumps((db_job_id, lda_desc, lda_output))[:150]

                else:
                    logging.debug("BAD JOB MATCH res {}".format(r))
                    logging.debug("job lda: {}".format(lda_job)[:300])
                    logging.debug("job db:  {}".format(db_job)[:300])
                    logging.debug("\n")

        else:
            logging.debug("BAD LENGTH MATCH res {}".format(r))
            dump_res_lda(res_lda)
            dump_res_db(res_db)
            logging.debug("\n")

        # logging.debug("\n")


def length_distrib(lists, max_len=20):
    length_counts = [0]*(max_len+1)
    for i, lst in enumerate(lists):
        if i % 100000 == 0:
            logging.debug("\t{}".format(i))
        length = min(len(lst), max_len)
        length_counts[length] += 1
    for j in range(len(length_counts)):
        logging.debug("length {}:\t{}".format(j, length_counts[j]))
    return length_counts



########################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default=None)
    parser.add_argument('--db', default=None)
    args = parser.parse_args()

    logging.info("connecting to db")
    conn = impdb.get_connection(args.host, args.db, args.user)

    # logging.debug("lda lengths:")
    # length_distrib(get_resumes_lda(LDA_FILE), 20)
    #
    # logging.debug("db lengths:")
    # length_distrib(get_resumes_db(conn), 20)

    # logging.info("marrying")
    # marry_lda_db(conn)

    res_dbs = get_resumes_db(conn)
    job_id_hash = get_job_id_hash(res_dbs)

    logging.debug("matching lda resumes to db hash")
    hits = 0
    misses = 0
    for r, res_lda in enumerate(get_resumes_lda(LDA_FILE)):
        if r % 10000 == 0:
            logging.debug("\t{}\t({} hits, {} misses)".format(r, hits, misses))

        key = make_resume_date_key_lda(res_lda)
        if key in job_id_hash:
            hits += 1
        else:
            misses += 1