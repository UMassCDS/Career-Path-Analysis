import argparse
import logging
import resume_import_db as impdb


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


##################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import resume data into relational db')
    parser.add_argument('host')
    parser.add_argument('user')
    parser.add_argument('db')
    parser.add_argument('--chunk_size', default=None)
    args = parser.parse_args()

    logging.info("connecting to db")
    conn = impdb.get_connection(args.host, args.db, args.user)

    logging.info("geocoding {} records".format(args.chunk_size if args.chunk_size else 1000000))
    impdb.geocode_blank_locs(conn, chunk_size=args.chunk_size)

    conn.commit()
