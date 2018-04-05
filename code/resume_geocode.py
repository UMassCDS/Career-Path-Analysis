import sys
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

STATE__ABBREV = {
    'ALABAMA': 'AL',
    'ALASKA': 'AK',
    'ARIZONA': 'AZ',
    'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA',
    'COLORADO': 'CO',
    'CONNECTICUT': 'CT',
    'DELAWARE': 'DE',
    'FLORIDA': 'FL',
    'GEORGIA': 'GA',
    'HAWAII': 'HI',
    'IDAHO': 'ID',
    'ILLINOIS': 'IL',
    'INDIANA': 'IN',
    'IOWA': 'IA',
    'KANSAS': 'KS',
    'KENTUCKY': 'KY',
    'LOUISIANA': 'LA',
    'MAINE': 'ME',
    'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA',
    'MICHIGAN': 'MI',
    'MINNESOTA': 'MN',
    'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO',
    'MONTANA': 'MT',
    'NEBRASKA': 'NE',
    'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH',
    'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM',
    'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND',
    'OHIO': 'OH',
    'OKLAHOMA': 'OK',
    'OREGON': 'OR',
    'PENNSYLVANIA': 'PA',
    'RHODE ISLAND': 'RI',
    'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD',
    'TENNESSEE': 'TN',
    'TEXAS': 'TX',
    'UTAH': 'UT',
    'VERMONT': 'VT',
    'VIRGINIA': 'VA',
    'WASHINGTON': 'WA',
    'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI',
    'WYOMING': 'WY',
    'GUAM': 'GU',
    'PUERTO RICO': 'PR',
    'VIRGIN ISLANDS': 'VI'
}
STATE_NAMES = set([k.lower() for k in STATE__ABBREV.keys()])
STATE_ABBREVS = set([v.lower() for v in STATE__ABBREV.values()])



# this gets called on all initial records, but not neccesarily fallbacks
def preprocess_loc(loc_str_raw):
    loc_str = loc_str_raw.strip().lower()

    # some dirty hacks to deal with a common geocoding errors
    ny_split = loc_str.split('new york')
    if len(ny_split) > 2:
        loc_str = "NY".join(loc_str.rsplit('new york', 1))

    # loc_str_elts = loc_str.split()
    #
    # if loc_str.endswith(' ca'):
    #     loc_str = loc_str.rsplit(' ca', 1)[0] + ' california United States of America'
    #
    # if loc_str == 'ny ny':
    #     loc_str = 'new york ny United States of America'
    #
    # if 'nyc' in loc_str_elts:
    #     loc_str = ' '.join(['new york city' if (elt == 'nyc') else elt for elt in loc_str_elts])
    #     loc_str += " United States of America"

    loc_str += ' United States of America'

    return loc_str


def geocode_loc(loc_str_raw, sleep_secs):

    loc_str = preprocess_loc(loc_str_raw)

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
        return loc_dict

    else:
        return None


def get_max_id(curs, table, id_col, cond):
    curs.execute("SELECT max(" + id_col + " ) FROM " + table + " WHERE " + cond)
    return curs.fetchone()[0]




def get_blank_locs_by_size(conn, chunk_size=None, match_str=None):
    curs = conn.cursor()
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
    return curs


def get_unamerican_locs_by_size(conn, chunk_size=None, match_str=None):
    curs = conn.cursor()
    logging.debug("selecting foreign locations to geocode")
    sql = "SELECT location, count(*) FROM " + JOB_TABLE + " "
    sql += "WHERE country<>'United States of America' "
    if match_str:
        sql += "AND location LIKE '" + match_str + "' "
    sql += "GROUP BY location ORDER BY count(*) DESC"
    if chunk_size:
        sql += " LIMIT " + str(chunk_size)
    logging.debug(sql)
    curs.execute(sql)
    return curs


def geocode_recs(conn, recs, fallback_funcs):
    logging.debug("updating records")
    curs_up = conn.cursor()
    fail_count = 0
    for r, rec in enumerate(recs):
        logging.debug("geocoding '{}'".format(rec))
        loc_str = rec[0]
        geo = geocode_loc(loc_str, GEOCODE_SLEEP_SECS)
        logging.debug("{} => {}".format(loc_str, geo))

        if geo is None:
            for fallback_func in fallback_funcs:
                loc_str_fallback = fallback_func(loc_str)
                if loc_str_fallback:
                    logging.debug("trying fallback '{}'".format(loc_str_fallback))
                    geo = geocode_loc(loc_str_fallback, GEOCODE_SLEEP_SECS)
                    logging.debug("{} => {}".format(loc_str, geo))
                    if geo:
                        break

        if geo:
            impdb.update_row(curs_up, JOB_TABLE, {'location': loc_str }, geo)
            fail_count = 0
            conn.commit()
        else:
            fail_count += 1
            if fail_count > GEOCODE_MAX_FAILS:
                raise Exception("{} geocode failures in a row".format(GEOCODE_MAX_FAILS))

    conn.commit()


state_names_abbrevs_sort = sorted(STATE_NAMES.union(STATE_ABBREVS),
                                  key=lambda x: len(x), reverse=True)
def unconcat_state(loc_str):
    for st in state_names_abbrevs_sort:
        if loc_str.endswith(st) and not loc_str.endswith(' ' + st):
            stuff, _ = loc_str.rsplit(st, 1)
            return preprocess_loc(stuff + ' ' + st + ' United States of America')
    return None


def unamerican_loc(loc_str):
    tag = 'United States of America'
    if loc_str.endswith(tag):
        return loc_str[:loc_str.index(tag)]
    else:
        return None


_compound_locs = None
def load_compound_locs(curs, num=20000):
    global _compound_locs

    logging.debug("fetching {} compound location names".format(num))
    sql = "SELECT location, count(*) FROM " + JOB_TABLE + " WHERE location LIKE '% % %' "
    sql += "GROUP BY location ORDER BY count(*) DESC LIMIT " + str(num)
    curs.execute(sql)

    _compound_locs = dict()
    for rec in curs:
        loc_str = rec[0].strip().lower()
        loc_elts = loc_str.split()
        loc_squished = loc_elts[0] + loc_elts[1]
        _compound_locs[loc_squished] = loc_elts[0] + ' ' + loc_elts[1]

    return curs.rowcount


def match_compound_names(loc_str):
    global _compound_locs

    loc_str = loc_str.strip().lower()
    loc_elts = loc_str.split()
    if loc_elts and (loc_elts[0] in _compound_locs):
        unsquished = _compound_locs[loc_elts[0]]
        loc_str_unsquished = ' '.join([unsquished] + loc_elts[1:])
        return preprocess_loc(loc_str_unsquished)
    return None


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
    # geocode_blank_locs_by_size(conn, chunk_size=args.chunk_size, match_str=args.match)

    # load_compound_locs(conn.cursor(), num=20000)
    # blank_recs = get_blank_locs_by_size(conn, chunk_size=args.chunk_size, match_str=args.match)
    # geocode_recs(conn, blank_recs, [match_compound_names])

    recs = get_unamerican_locs_by_size(conn, chunk_size=args.chunk_size, match_str=args.match)
    geocode_recs(conn, recs, [unamerican_loc])

    conn.commit()



