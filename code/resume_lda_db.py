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


def marry_lda_db(conn):
    logging.info("loading lda resumes")
    # res_ldas = resume_lda.load_json_resumes_lda(LDA_FILE, min_len=0)
    res_dbs = get_resumes_db(conn)

    # for res_lda in res_ldas:
    for res_lda in get_resumes_lda(LDA_FILE):
        for j, job in enumerate(res_lda):
            logging.debug("res lda ({}/{}): {}".format(j, len(res_lda)-1, job)[:200])

        res_db = res_dbs.next()
        for j, job in enumerate(res_db):
            logging.debug("res db  ({}/{}): {}".format(j, len(res_db)-1, job)[:200])

        if len(res_lda) == len(res_db):
            for job_lda, job_db in zip(res_lda, res_db):
                logging.debug("job lda: {}".format(job_lda)[:200])
                logging.debug("job db:  {}".format(job_db)[:200])
        else:
            logging.debug("BAD LENGTH MATCH!")

    logging.debug("\n")



########################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default=None)
    parser.add_argument('--db', default=None)
    args = parser.parse_args()

    logging.info("connecting to db")
    conn = impdb.get_connection(args.host, args.db, args.user)

    logging.info("marrying")
    marry_lda_db(conn)
