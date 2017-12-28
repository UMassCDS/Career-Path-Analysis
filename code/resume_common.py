import collections
import json
import datetime


DATE_FORMAT = "%Y%m%d"


# each individual resume is a list of ResumeEntry objects
ResumeEntry = collections.namedtuple('ResumeEntry', ['start', 'end', 'company', 'desc'])


def sort_stints(resume):
    stints_sorted = sorted([ s for s in resume if ((s[0] is not None) or (s[1] is not None)) ],
                           key=lambda x: x[0] or x[1])
    # sys.stderr.write("stints sort: {}\n".format(stints_sorted))

    if len(stints_sorted) > 0:
        # first start and last end are special cases
        s, e, c, d = stints_sorted[0]
        if s is None:
            stints_sorted[0] = (e, e, c, d)
        s, e, c, d = stints_sorted[-1]
        if e is None:
            stints_sorted[-1] = (s, s, c, d)

        # now fill in the rest
        all_dates = sorted([ s[0] for s in stints_sorted if s[0] is not None ] +
                           [ s[1] for s in stints_sorted if s[1] is not None ])
        stints_filled = []
        for start, end, co, desc in stints_sorted:
            if end is None:
                ends = [ c for c in all_dates if c > start ]
                end = ends[0] if len(ends) > 0 else start
            elif start is None:
                starts = [ c for c in all_dates if c < end ]
                start = starts[-1] if len(starts) > 0 else end
            stints_filled.append(ResumeEntry(start, end, co, desc))
        # sys.stderr.write("stints fill: {}\n".format(stints))
        stints_sorted = stints_filled
    return stints_sorted


def tuplify(res_ent):
    print "ow ", res_ent
    s, e, c, d = tuple(res_ent)
    return (s.strftime(DATE_FORMAT), e.strftime(DATE_FORMAT), c, d)


def detuplify(res_tup):
    s, e, c, d = res_tup
    return ResumeEntry(datetime.datetime.strptime(s, DATE_FORMAT).date(),
                       datetime.datetime.strptime(e, DATE_FORMAT).date(),
                       c, d)


# def transform_descs_vector(resume_list):
#     descs, job_sequence_counts = get_descs_flat(resume_list)
#     termfreq_vectorizer = CountVectorizer()
#     termfreqs = termfreq_vectorizer.fit_transform()
#     print "vectorized {} resumes into {} matrix".format(len(descs), termfreqs.shape)
#     descs_vector = unflatten(termfreqs, job_sequence_counts)
#     return descs_vector





