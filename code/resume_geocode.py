import argparse
import logging
import time

import geopy.exc
import geopy.geocoders

import resume_import_db as impdb
from resume_import_db import JOB_TABLE

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

GEOCODE_SLEEP_SECS = 1
GEOCODE_ATTEMPTS = 5
GEOCODE_MAX_FAILS = 100

_geolocator = geopy.geocoders.Nominatim()
_geocode_cache = {}
_geocode_cache_hits = 0
_geocode_cache_misses = 0


def geocode_loc(loc_str_raw, sleep_secs):
    global _geolocator, _geocode_cache, _geocode_cache_hits, _geocode_cache_misses

    loc_str = loc_str_raw.strip().lower()

    # some dirty hacks to deal with a common geocoding errors
    if loc_str.endswith(' ca'):
        loc_str = loc_str.rsplit(' ca', 1)[0] + ' california'

    loc_str_elts = loc_str.split()
    # if 'ny' in loc_str_elts:
    #     logging.debug(loc_str + " => ")
    #     loc_str = ' '.join(['new york' if (elt == 'ny') else elt for elt in loc_str_elts])
    #     logging.debug(loc_str)
    if loc_str == 'ny ny':
        loc_str = 'new york ny'
    elif 'nyc' in loc_str_elts:
        loc_str = ' '.join(['new york city' if (elt == 'nyc') else elt for elt in loc_str_elts])

    if loc_str in _geocode_cache:
        _geocode_cache_hits += 1
        return _geocode_cache[loc_str]
    else:
        _geocode_cache_misses += 1

    # if (_gecode_cache_hits + _gecode_cache_misses) % 100 == 0:
    #     geocode_cache_report()

    for attempt in range(GEOCODE_ATTEMPTS):
        try:
            location = _geolocator.geocode(loc_str, exactly_one=True, timeout=10)
            if sleep_secs:
                time.sleep(sleep_secs)
            break
        except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderUnavailable) as err:
            logging.debug("geocode failure {}: {}".format(attempt, err))
            time.sleep(30*(attempt+1))
            continue
    else:
        # raise Exception("{} geocode failure attempts in a row".format(GEOCODE_ATTEMPTS))
        logging.debug("{} geocode failure attempts in a row".format(GEOCODE_ATTEMPTS))
        return None

    if location:
        # location = get_best_geo(locations)

        # Nominatim format is ...city, (county,) state, (zip,) country
        addr_elts = location.address.split(', ')

        # if there's a zip it'll be second to last
        if (len(addr_elts) >= 2) and addr_elts[-2].isnumeric():
            # logging.debug(addr_elts)
            zip = addr_elts.pop(-2)

        num_elts = len(addr_elts)
        if num_elts > 3:  # got a county and city
            city, county, state, country = addr_elts[-4:]
        elif num_elts == 3:
            city, state, country = addr_elts[-3:]
        elif num_elts == 2:
            city, country = addr_elts[-2:]
            state = None
        elif num_elts == 1:
            country = addr_elts[0]
            city = None
            state = None
        else:
            logging.debug("empty geoloc address: " + location.address)
            country = None
            city = None
            state = None

        loc_dict = {'city': city, 'state': state, 'country': country,
                    'latitude':location.latitude, 'longitude': location.longitude}
        _geocode_cache[loc_str] = loc_dict
        # logging.debug("geocoded {} => {}".format(loc_str_raw, loc_dict))
        return loc_dict

    else:
        _geocode_cache[loc_str] = None
        return None


def geocode_cache_report():
    h = _geocode_cache_hits
    m = _geocode_cache_misses
    perc = float(h)/(h+m) if (h+m) > 0 else 0
    logging.debug("geocode cache: {} entries, {} calls ({} hits, {} misses, {})".format(
            len(_geocode_cache), h + m, h, m, perc))


def get_max_id(curs, table, id_col, cond):
    curs.execute("SELECT max(" + id_col + " ) FROM " + table + " WHERE " + cond)
    return curs.fetchone()[0]


def geocode_blank_locs(conn, chunk_size=None, cont=True, match_str=None):
    global _geocode_cache
    curs = conn.cursor()

    # get the id of the last record we successfully geocoded
    last_id = -1
    if cont:
        # curs.execute("SELECT max(job_id) FROM " + JOB_TABLE + " WHERE country IS NOT NULL")
        # last_id = curs.fetchone()[0]
        last_id = get_max_id(curs, JOB_TABLE, 'job_id', 'country IS NOT NULL')
        logging.debug("last known geocoded job_id: {}".format(last_id))
        if last_id is not None:
            load_cache(curs, max_id=last_id)

    else:
        load_cache(curs)

    # now update the cache with the ones that we got already
    # logging.debug("updating geocode cache")
    # sql = "SELECT DISTINCT location, city, state, country, latitude, longitude "
    # sql += "FROM " + JOB_TABLE + " "
    # sql += "WHERE job_id <= %s AND location IS NOT NULL"
    # curs.execute(sql, (last_id,))
    # for rec in curs:
    #     loc_str_raw, city, state, country, latitude, longitude = rec
    #     # n.b.: make sure this matches cacheing above
    #     loc_str = loc_str_raw.strip().lower()
    #     loc_dict = {'city': city, 'state': state, 'country': country,
    #                 'latitude': latitude, 'longitude': longitude}
    #     _geocode_cache[loc_str] = loc_dict
    # logging.debug("cached {} locations from {} rows".format(len(_geocode_cache), curs.rowcount))

    # grab the records to be updated
    logging.debug("selecting records to geocode")
    sql = "SELECT job_id, location FROM " + JOB_TABLE + " WHERE job_id > %s "
    sql += "AND location IS NOT NULL AND country IS NULL "
    if match_str:
        sql += "AND location LIKE '" + match_str + "' "
    sql += "ORDER BY job_id"
    if chunk_size:
        sql += " LIMIT " + str(chunk_size)

    logging.debug(sql.replace('%s', '{}').format(last_id))
    logging.debug(sql)
    logging.debug(last_id)
    curs.execute(sql, (last_id,))

    logging.debug("updating {} records".format(curs.rowcount))
    curs_up = conn.cursor()
    fail_count = 0
    for r, rec in enumerate(curs):
        job_id, loc_str = rec
        geo = geocode_loc(loc_str, GEOCODE_SLEEP_SECS)
        logging.debug("{} => {}".format(loc_str, geo))

        # curs_up.execute("SELECT count(*) FROM " + JOB_TABLE  + " WHERE country IS NULL")
        # logging.debug("before update: {} null country records".format(curs_up.fetchone()[0]))

        if geo:
            impdb.update_row(curs_up, JOB_TABLE, {'job_id': job_id}, geo)
            fail_count = 0

            conn.commit()
            # curs_up.execute("SELECT count(*) FROM " + JOB_TABLE  + " WHERE country IS NULL")
            # logging.debug("after update: {} null country records".format(curs_up.fetchone()[0]))

        else:
            fail_count += 1
            if fail_count > GEOCODE_MAX_FAILS:
                raise Exception("{} geocode failures in a row".format(GEOCODE_MAX_FAILS))

        if r % 100 == 0:
            logging.debug("geocode updated {}/{} records".format(r, curs.rowcount))
            geocode_cache_report()
            conn.commit()

    conn.commit()


def geocode_blank_locs_by_size(conn, chunk_size=None, match_str=None):
    global _geocode_cache
    curs = conn.cursor()

    load_cache(curs)

    # grab the locations to geocoded
    logging.debug("selecting locations to geocode")
    sql = "SELECT location, count(*) FROM " + JOB_TABLE + " "
    sql += "WHERE location IS NOT NULL AND country IS NULL "
    if match_str:
        sql += "AND location LIKE '" + match_str + "' "
    sql += "GROUP BY location ORDER BY count(*) DESC"
    if chunk_size:
        sql += " LIMIT " + str(chunk_size)
    logging.debug(sql)
    curs.execute(sql)

    logging.debug("updating records")
    curs_up = conn.cursor()
    fail_count = 0
    for r, rec in enumerate(curs):
        loc_str, rec_count = rec
        geo = geocode_loc(loc_str, GEOCODE_SLEEP_SECS)
        logging.debug("{} => {}".format(loc_str, geo))

        # curs_up.execute("SELECT count(*) FROM " + JOB_TABLE  + " WHERE country IS NULL")
        # logging.debug("before update: {} null country records".format(curs_up.fetchone()[0]))

        if geo:
            impdb.update_row(curs_up, JOB_TABLE, {'location': loc_str}, geo)
            fail_count = 0

            conn.commit()
            # curs_up.execute("SELECT count(*) FROM " + JOB_TABLE  + " WHERE country IS NULL")
            # logging.debug("after update: {} null country records".format(curs_up.fetchone()[0]))

        else:
            fail_count += 1
            if fail_count > GEOCODE_MAX_FAILS:
                raise Exception("{} geocode failures in a row".format(GEOCODE_MAX_FAILS))

        if r % 100 == 0:
            logging.debug("geocode updated {}/{} records".format(r, curs.rowcount))
            geocode_cache_report()
            conn.commit()

    conn.commit()


def load_cache(curs, min_id=None, max_id=None):
    params = []
    logging.debug("updating geocode cache")
    sql = "SELECT DISTINCT location, city, state, country, latitude, longitude "
    sql += "FROM " + JOB_TABLE + " "
    sql += "WHERE location IS NOT NULL AND country IS NOT NULL"
    if min_id:
        sql += " AND job_id >= %s"
        params.append(min_id)
    if max_id:
        sql += " AND job_id <= %s"
        params.append(max_id)

    logging.debug(sql.replace('%s', '{}').format(*params))
    curs.execute(sql, params)
    for rec in curs:
        loc_str_raw, city, state, country, latitude, longitude = rec
        # n.b.: make sure this matches cacheing above
        loc_str = loc_str_raw.strip().lower()
        loc_dict = {'city': city, 'state': state, 'country': country,
                    'latitude': latitude, 'longitude': longitude}
        _geocode_cache[loc_str] = loc_dict
    logging.debug("cached {} locations from {} rows".format(len(_geocode_cache), curs.rowcount))
    return curs.rowcount



##################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import resume data into relational db')
    parser.add_argument('host')
    parser.add_argument('user')
    parser.add_argument('db')
    parser.add_argument('--chunk_size', default=None)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--match', default=None)
    args = parser.parse_args()

    logging.info("connecting to db")
    conn = impdb.get_connection(args.host, args.db, args.user)

    logging.info("geocoding {} records".format(args.chunk_size if args.chunk_size else 1000000))
    # geocode_blank_locs(conn, chunk_size=args.chunk_size, cont=args.cont, match_str=args.match)
    geocode_blank_locs_by_size(conn, chunk_size=args.chunk_size, match_str=args.match)

    conn.commit()
