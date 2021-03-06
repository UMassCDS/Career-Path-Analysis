import re
import getpass
import argparse
import logging
import psycopg2 as db
import resume_import


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

# DB_NAME = 'careerpaths'

RESUME_TABLE = 'resumes'
RESUME_COLS = [
    ('resume_id', 'INTEGER PRIMARY KEY'),
    ('resume_title', 'TEXT'),
    ('desired_job_title', 'VARCHAR(256)'),
    ('state', 'VARCHAR(256)'),
    ('salary', 'FLOAT'),
    ('currency_name', 'VARCHAR(8)'),
    ('summary', 'TEXT'),
    ('additional_info', 'TEXT'),
    ('will_relocate', 'BOOLEAN'),
    ('commute_distance', 'INTEGER'),
    ('willingness_to_travel_internal_name', 'VARCHAR(32)'),
    ('wants_permanent', 'BOOLEAN'),
    ('wants_contract', 'BOOLEAN'),
    ('wants_intern', 'BOOLEAN'),
    ('wants_temp', 'BOOLEAN'),
    ('wants_full_time', 'BOOLEAN'),
    ('wants_part_time', 'BOOLEAN'),
    ('wants_per_diem', 'BOOLEAN'),
    ('wants_overtime', 'BOOLEAN'),
    ('wants_weekends', 'BOOLEAN'),
    ('prefer_weekends', 'BOOLEAN'),
    ('wants_seasonal', 'BOOLEAN'),
    ('date_created', 'DATE'),
    ('date_modified', 'DATE')
]

JOB_TABLE = 'jobs'
JOB_COLS = [
    ('job_id', 'SERIAL PRIMARY KEY'),
    ('resume_id', 'INTEGER'),
    ('start_dt', 'DATE'),
    ('end_dt', 'DATE'),
    ('company_name', 'VARCHAR(256)'),
    ('location', 'VARCHAR(256)'),
    ('title', 'VARCHAR(256)'),
    ('description', 'TEXT'),
    ('city', 'VARCHAR(192)'),
    ('state', 'VARCHAR(64)'),
    ('country', 'VARCHAR(64)'),
    ('latitude', 'FLOAT'),
    ('longitude', 'FLOAT')
]

EDU_TABLE = 'schools'
EDU_COLS = [
    ('edu_id', 'SERIAL PRIMARY KEY'),
    ('resume_id', 'INTEGER'),
    ('school_id', 'INTEGER'),
    ('school_name', 'VARCHAR(256)'),
    ('city', 'VARCHAR(256)'),
    ('state', 'VARCHAR(64)'),
    ('grad_date', 'DATE'),
    ('gpa', 'FLOAT'),
    ('summary', 'TEXT')
]


# https://www.dataquest.io/blog/loading-data-into-postgres/


def parse_all_resumes(conn, infile_names):
    curs = conn.cursor()
    resume_count = 0
    err_count = 0
    for i, resume_xml in enumerate(resume_import.get_resume_xmls(infile_names)):
        resume_count += 1

        if i % 1000 == 0:
            logging.debug("resume xml {}".format(i))

        ret = parse_resume_db(curs, resume_xml)
        if ret:
            err_count += 1
        if i % 1000 == 0:
            conn.commit()

    if err_count > 0:
        logging.warning("encountered {} errors while loading {} resumes".format(err_count,
                                                                                resume_count))


# 		<ResumeID>82408838</ResumeID>
# 		<WantsPermanent>True</WantsPermanent>
# 		<WantsContract>False</WantsContract>
# 		<WantsIntern>False</WantsIntern>
# 		<WantsTemp>False</WantsTemp>
# 		<WantsFullTime>True</WantsFullTime>
# 		<WantsPartTime>False</WantsPartTime>
# 		<WantsPerDiem>False</WantsPerDiem>
# 		<WantsOvertime>False</WantsOvertime>
# 		<WantsWeekends>False</WantsWeekends>
# 		<PreferWeekends>False</PreferWeekends>
# 		<WantsSeasonal>False</WantsSeasonal>
# 		<Salary>50000.0000</Salary>
# 		<CurrencyName>USD</CurrencyName>
# 		<CommuteDistance>0</CommuteDistance>
# 		<RelocationComments/>
# 		<AdditionalInfo/>
# 		<ResumeTitle>general manager</ResumeTitle>
# 		<DateCreated>1/16/2006 12:00:00 AM</DateCreated>
# 		<DateModified>4/3/2006 7:33:00 PM</DateModified>
# 		<WillRelocate>0</WillRelocate>
# 		<Summary/>
# 		<ChannelName>New Monster</ChannelName>
# 		<WillingnessToTravelInternalName>Up to 25% travel</WillingnessToTravelInternalName>
# 		<DesiredJobTitle>General Manager</DesiredJobTitle>
def parse_resume_db(curs, resume_xml):
    attrs = {}

    # First, grab all the resume fields

    resume_id = int(resume_xml.findtext('ResumeID'))
    attrs['resume_id'] = resume_id

    for attr_name in ['WantsPermanent',
                      'WantsContract',
                      'WantsIntern',
                      'WantsTemp',
                      'WantsFullTime',
                      'WantsPartTime',
                      'WantsPerDiem',
                      'WantsOvertime',
                      'WantsWeekends',
                      'PreferWeekends',
                      'WantsSeasonal',
                      'WillRelocate']:
        attrs[decamel(attr_name)] = boolnone(resume_xml.findtext(attr_name))

    for attr_name in ['CurrencyName',
                      'AdditionalInfo',
                      'ResumeTitle',
                      'Summary',
                      'WillingnessToTravelInternalName',
                      'DateCreated',
                      'DateModified']:
        attrs[decamel(attr_name)] = resume_xml.findtext(attr_name)

    attrs['desired_job_title'] = resume_import.clean_name(resume_xml.findtext('DesiredJobTitle'))

    sal = floatnone(resume_xml.findtext('Salary'))
    # this is messed up
    # if (sal is not None) and (sal < 1000):
    #     sal *= 40*52
    attrs['salary'] = sal

    attrs['commute_distance'] = intnone(resume_xml.findtext('CommuteDistance'))

    states = resume_import.get_states(resume_xml)
    attrs['state'] = ','.join(states) if states else None

    # ignored fields:
    #   'RelocationComments'
    #   'ChannelName'

    try:
        insert_row(curs, RESUME_TABLE, attrs)
    except Exception as err:
        logging.warning("error inserting row: {}".format(err))

        # if we can't insert the resume row, don't bother with the others
        return False

    # Now look through job experience entries
    # experiences = []
    experience = resume_xml.find('experience')
    if experience is not None:
        stints = []
        for exprec in experience:
            title = resume_import.clean_name(exprec.findtext('Title'), None)
            company_name = resume_import.get_company_name(exprec, False)
            start = resume_import.make_date(exprec.find('StartYear'), exprec.find('StartMonth'))
            end = resume_import.make_date(exprec.find('EndYear'), exprec.find('EndMonth'))
            desc = resume_import.get_description(exprec)
            location = resume_import.clean_name(exprec.findtext('Location'), None)

            stints.append((start, end, company_name, location, title, desc))

        for start, end, company_name, location, title, description in sort_stints(stints):
            exp_attrs = { 'resume_id': resume_id,
                          'start_dt': start,
                          'end_dt': end,
                          'company_name': company_name,
                          'location': location,
                          'title': title,
                          'description': description
                        }
            insert_row(curs, JOB_TABLE, exp_attrs)
            # experiences.append(exp_attrs)


    # <schoolrecord>
    # 		<City>Glendale</City>
    # 		<CompleteMonth>5</CompleteMonth>
    # 		<CompleteYear>1977</CompleteYear>
    # 		<EducationSubject/>
    # 		<EducationSummary/>
    # 		<GPA>0.00</GPA>
    # 		<School>Glendale Community College</School>
    # 		<SchoolId>3359</SchoolId>
    # 		<State>Arizona</State>
    # 		<DisplaySort>110</DisplaySort>
    # 	</schoolrecord>
    education = resume_xml.find('education')
    if education is not None:
        for schoolrecord in education:
            edu_attrs = dict()
            edu_attrs['resume_id'] = resume_id
            edu_attrs['grad_date'] = resume_import.make_date(schoolrecord.find('CompleteYear'),
                                                schoolrecord.find('CompleteMonth'))
            edu_attrs['school_name'] = resume_import.clean_name(schoolrecord.findtext('School'))
            edu_attrs['school_id'] = schoolrecord.findtext('SchoolId')
            edu_attrs['city'] = resume_import.clean_name(schoolrecord.findtext('City'))
            edu_attrs['state'] = resume_import.clean_name(schoolrecord.findtext('State'))
            edu_attrs['summary'] = schoolrecord.findtext('EducationSummary')
            edu_attrs['gpa'] = schoolrecord.findtext('GPA')
            insert_row(curs, EDU_TABLE, edu_attrs)

    return 0


def sort_stints(resume):
    # we expect tups of (start, end, ...)
    stints_sorted = sorted([ s for s in resume if ((s[0] is not None) or (s[1] is not None)) ],
                           key=lambda x: x[0] or x[1])
    # sys.stderr.write("stints sort: {}\n".format(stints_sorted))

    if len(stints_sorted) > 0:
        # first start and last end are special cases
        # s, e, c, d = stints_sorted[0]
        s = stints_sorted[0][0]
        e = stints_sorted[0][1]
        rest = stints_sorted[0][2:]
        if s is None:
            stints_sorted[0] = (e, e) + rest

        # s, e, c, d = stints_sorted[-1]
        s = stints_sorted[-1][0]
        e = stints_sorted[-1][1]
        rest = stints_sorted[-1][2:]
        if e is None:
            stints_sorted[-1] = (s, s) + rest

        # now fill in the rest
        all_dates = sorted([ s[0] for s in stints_sorted if s[0] is not None ] +
                           [ s[1] for s in stints_sorted if s[1] is not None ])
        stints_filled = []
        # for start, end, co, desc in stints_sorted:
        for stint in stints_sorted:
            start = stint[0]
            end = stint[1]
            rest = stint[2:]
            if end is None:
                ends = [ c for c in all_dates if c > start ]
                end = ends[0] if len(ends) > 0 else start
            elif start is None:
                starts = [ c for c in all_dates if c < end ]
                start = starts[-1] if len(starts) > 0 else end
            # stints_filled.append(ResumeEntry(start, end, co, desc))
            stints_filled.append((start, end) + rest)
        # sys.stderr.write("stints fill: {}\n".format(stints))
        stints_sorted = stints_filled
    return stints_sorted


def find_one(lst, targets):
    for i in range(len(lst)-1, -1, -1):
        if lst[i] in targets:
            return i
    return -1


def standardize_job_locations(conn):
    curs = conn.cursor()
    curs.execute("SELECT job_id, location FROM " + JOB_TABLE + " WHERE location IS NOT NULL")

    state_names = set(STATE__ABBREV.keys())
    state_abbrevs = set(STATE__ABBREV.values())
    state_names_abbrevs = state_names.union(state_abbrevs)

    state_twos = set([ tuple(st.split()) for st in state_names if st.find(' ')>=0 ])  # two word states
    state_twos_first = set([ st[0] for st in state_twos ])  # first word of above

    print state_twos
    print state_twos_first

    curs_up = conn.cursor()

    good_count = 0
    bad_count = 0
    for r, row in enumerate(curs):
        job_id, loc_raw = row
        # most common format: city, state
        loc_elts = loc_raw.strip().upper().split()

        if loc_elts == []:
            continue

        # do a little dance to deal with two-word states
        pos = find_one(loc_elts, state_twos_first)
        if pos > -1:
            # print "found state_twos_first {} in ".format(pos), loc_elts
            if pos+1 <= len(loc_elts)-1:
                # print "yup"
                if tuple(loc_elts[pos:pos+2]) in state_twos:
                    loc_elts = loc_elts[:pos] + [" ".join(loc_elts[pos:pos+2])] + loc_elts[pos+2:]
                    # print "loc_elts now", loc_elts

        state_pos = find_one(loc_elts, state_names_abbrevs)
        # if state_pos == -1:
        #     # last chance for locs like "old hickorytn"
        #     loc_elts_last = loc_elts[-1]
        #     for st in state_names_abbrevs:
        #         if loc_elts_last.endswith(st):
        #             print loc_elts
        #             stuff, _ = loc_elts_last.rsplit(st, 1)
        #             loc_elts = loc_elts[:-1] + [stuff, st]
        #             state_pos = len(loc_elts) - 1
        #             break

        if state_pos == -1:
            print "no state: " + loc_raw
            bad_count += 1
            continue
        state = loc_elts[state_pos]
        if state in STATE__ABBREV:
            state = STATE__ABBREV[state]
        country = " ".join(loc_elts[state_pos+1:])
        city = " ".join(loc_elts[:state_pos])
        good_count += 1
        # print "{} => {}, {}, {}".format(loc_raw, city, state, country)

        curs_up.execute("UPDATE " + JOB_TABLE + " SET city=%s, state=%s WHERE job_id=%s",
                         (city, state, job_id))
        if r % 10000 == 0:
            print "committing {}".format(r)
            conn.commit()

    print "{} good, {} bad ({})".format(good_count, bad_count,
                                        float(good_count)/(good_count+bad_count))


# # Right now, just pick the first one that's in the USA (this method is essentially here to stop
# # 'ca' from getting turned into 'Candada' instead of 'California'
# def get_best_geo(locations):
#     if len(locations) == 1:
#         return locations[0]
#     else:
#         logging.debug("got {} locs: {}".format(len(locations), [loc.address for loc in locations]))
#
#     priority_rank = {
#         'United States of America': 0,
#         'Canada': 1
#     }
#     locations.sort(key=lambda loc: priority_rank.get(loc.address.rsplit(', ', 1)[-1], sys.maxint))
#     return locations[0]


def add_column(conn, tab_name, col_name, col_type):
    curs = conn.cursor()
    sql = "ALTER TABLE {} ADD COLUMN {} {}".format(tab_name, col_name, col_type)
    print sql
    curs.execute(sql)
    conn.commit()


def get_connection(host, dbname, user=None):
    if user is None:
        user = raw_input("db username: ")
    passw = getpass.getpass("db password: ")
    connstr = "host='{}' dbname={} user={} password='{}'".format(host, dbname, user, passw)
    return  db.connect(connstr)
# _connection = None
# _dbhost = None
# def get_connection(u=None):
#     global _connection
#     global _dbhost
#
#     if not _connection:
#         user = raw_input("db username: ")
#         passw = getpass.getpass("db password: ")
#         connstr = "host='{}' dbname={} user={} password='{}'".format(_dbhost, DB_NAME, user, passw)
#         _connection = db.connect(connstr)
#     return _connection
#
#
# _cursor = None
# def get_cursor():
#     global _cursor
#     if not _cursor:
#         _cursor = get_connection().cursor()
#     return _cursor


def create_all_tables(conn, overwrite=False):
    curs = conn.cursor()
    if overwrite:
        drop_table(curs, RESUME_TABLE)
        drop_table(curs, JOB_TABLE)
        drop_table(curs, EDU_TABLE)
    create_table(curs, RESUME_TABLE, RESUME_COLS)
    create_table(curs, JOB_TABLE, JOB_COLS)
    create_table(curs, EDU_TABLE, EDU_COLS)


def create_table(curs, table_name, col_tups):
    sql = "CREATE TABLE " + table_name + " ("
    sql += ", ".join([nam + " " + typ for nam, typ in col_tups])
    sql += ")"
    curs.execute(sql)


def drop_table(curs, table_name):
    sql = "DROP TABLE IF EXISTS " + table_name
    curs.execute(sql)


def insert_row(curs, table_name, col__val):
    col_val_tups = col__val.items()
    cols = [ t[0] for t in col_val_tups ]
    vals = [ t[1] for t in col_val_tups ]
    sql = "INSERT INTO " + table_name + " ("
    sql += ", ".join(cols)
    sql += ") VALUES ("
    sql += ", ".join(["%s"]*len(vals))
    sql += ")"
    try:
        curs.execute(sql, vals)
    except Exception as err:
        logging.warning("error inserting record: {} {}".format(sql, vals))
        raise err


def update_row(curs, table_name, where_dict, update_dict):
    # where_names, where_vals = zip(*where_dict.items())
    up_names, up_vals = zip(*update_dict.items())
    sql = "UPDATE " + table_name
    sql += " SET " + ", ".join([u + "=%s" for u in up_names])
    # sql += " WHERE " + " AND ".join([w + "=%s" for w in where_names])

    sql += " WHERE "
    where_names = []
    where_vals = []
    where_nulls = []
    for name, val in where_dict.items():
        if val is not None:
            where_names.append(name)
            where_vals.append(val)
        else:
            where_nulls.append(name)
    sql += " AND ".join([w + "=%s" for w in where_names] + [w + " IS NULL" for w in where_nulls])

    logging.debug(sql.replace('%s', '{}').format(*(list(up_vals) + where_vals)))
    try:
        curs.execute(sql, list(up_vals) + where_vals)
        logging.debug("updated {} record(s)".format(curs.rowcount))

    except Exception as err:
        logging.debug("error updating record: '{} {}', {}".format(sql, list(up_vals) + where_vals, err))
        # raise err


# def commit():
#     get_connection().commit()


def boolnone(txt):
    return None if txt is None else ((txt.lower() == 'true') or (txt == '1'))


def floatnone(txt):
    return None if txt is None else float(txt)


def intnone(txt):
    return None if txt is None else int(txt)


def decamel(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()





#####################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import resume data into relational db')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default=None)
    parser.add_argument('--db', default=None)
    parser.add_argument('infile_names', nargs='+')
    args = parser.parse_args()

    # _dbhost = args.host

    logging.debug("got {} input files: \n\t{}".format(len(args.infile_names),
                                                      "\n\t".join(args.infile_names)))

    logging.info("connecting to db")
    conn = get_connection(args.host, args.db, args.user)

    logging.info("creating tables")
    create_all_tables(conn, overwrite=True)

    logging.info("loading infiles")
    # resume_import.xml2resumes(args.infile_names, parse_resume_db)
    parse_all_resumes(conn, args.infile_names)

    # sys.stderr.write("read {} resumes\n".format(len(resumes)))

    conn.commit()








    # resumes_clean = clean_resume_descs(resumes)
    # sys.stderr.write("cleaned {} resumes\n".format(len(resumes_clean)))
    #
    # # with open(out, 'wb') as outp:
    # #     pickle.dump(resumes_clean, outp)
    # dump_json_resumes(resumes_clean, out)


