import sys
import re
import psycopg2 as db
import resume_import
import resume_common



RESUME_COLS = [
    ('resume_id', 'INTEGER'),
    ('wants_permanent', 'BOOLEAN'),
    ('wants_contract', 'BOOLEAN'),
    ('wants_intern>', 'BOOLEAN'),
    ('wants_temp>', 'BOOLEAN'),
    ('wants_fullTime', 'BOOLEAN'),
    ('wants_partTime', 'BOOLEAN'),
    ('wants_perDiem', 'BOOLEAN'),
    ('wants_overtime', 'BOOLEAN'),
    ('wants_weekends', 'BOOLEAN'),
    ('prefer_weekends', 'BOOLEAN'),
    ('wants_seasonal', 'BOOLEAN'),
    ('will_relocate', 'BOOLEAN'),
    ('currency_name', 'VARCHAR(8)'),
    ('additional_info', 'TEXT'),
    ('resume_title', 'TEXT'),
    ('summary', 'TEXT'),
    ('willingness_to_travel_internal_name', 'VARCHAR(32)'),
    ('date_created', 'DATE'),
    ('date_modified', 'DATE'),
    ('desired_job_title', 'VARCHAR(64)'),
    ('salary', 'FLOAT')
    ('commute_distance', 'INTEGER')
    ('state', 'VARCHAR(16)')
]

JOB_COLS = [
    ('resume_id', 'INTEGER'),
    ('start', 'DATE'),
    ('end', 'DATE'),
    ('company_name', 'VARCHAR(64)'),
    ('location', 'VARCHAR(128)'),
    ('title', 'VARCHAR(64)'),
    ('description', 'TEXT)')
]



# https://www.dataquest.io/blog/loading-data-into-postgres/








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
def parse_resume_db(resume_xml):
    attrs = {}

    # First, grab all the resume fields

    resume_id = int(resume_xml.findtext('ResumeID'))
    attrs['resume_id'] = resume_id

    for attr_name in ['WantsPermanent',
                      'WantsContract',
                      'WantsIntern>',
                      'WantsTemp>',
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
                      'DateModified',
                      'DesiredJobTitle']:
        attrs[decamel(attr_name)] = resume_xml.findtext(attr_name)

    sal = floatnone(resume_xml.findtext('Salary'))
    if (sal is not None) and (sal < 1000):
        sal *= 40*52
    attrs['salary'] = sal

    attrs['commute_distance'] = intnone(resume_xml.findtext('CommuteDistance'))

    states = resume_import.get_states(resume_xml)
    attrs['state'] = ','.join(states) if states else None

    # ignored fields:
    #   'RelocationComments'
    #   'ChannelName'

    # Now look through job experience entries
    experiences = []
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
                          'start': start,
                          'end': end,
                          'company_name': company_name,
                          'location': location,
                          'title': title,
                          'description': description }
            experiences.append(exp_attrs)
    attrs['experience'] = experiences

    # education = resume.find('education')
    # if education is not None:
    #     for schoolrecord in education:
    #         grad_date = make_date(schoolrecord.find('CompleteYear'), schoolrecord.find('CompleteMonth'))
    #         school_name = schoolrecord.find('School').text if schoolrecord.find('School').text is not None else ""
    #         school_id = schoolrecord.find('SchoolId').text if schoolrecord.find('SchoolId').text is not None else ""
    #         school = unidecode.unidecode("EDU " + school_id + " " + school_name)
    #         stints.append((None, grad_date, school))

    return attrs




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


def resume_to_table(attrs):


def job_entry_to_table(attrs):


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
    USAGE = " usage: " + sys.argv[0] + " infile0.tgz [infile1.tgz infile2.tgz ...] outfile.json"
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    ins = sys.argv[1:-1]
    out = sys.argv[-1]

    sys.stderr.write(str(ins) + "\n")
    resumes = resume_import.xml2resumes(ins, parse_resume_db)
    sys.stderr.write("read {} resumes\n".format(len(resumes)))








    # resumes_clean = clean_resume_descs(resumes)
    # sys.stderr.write("cleaned {} resumes\n".format(len(resumes_clean)))
    #
    # # with open(out, 'wb') as outp:
    # #     pickle.dump(resumes_clean, outp)
    # dump_json_resumes(resumes_clean, out)

















