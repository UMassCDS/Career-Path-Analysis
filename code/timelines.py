import sys
import string
import datetime
import gzip
import pickle
import xml.etree.ElementTree as ET

import nltk
import nltk.collocations
from nltk.corpus import stopwords


BIGRAM_DELIM = '_'


def get_company_name(experiencerecord):
    company_name = experiencerecord.find('Company').text
    location = experiencerecord.find('Location').text if experiencerecord.find('Location') is not None else None

    ret = clean_name(company_name)
    if (location is not None) and (len(location.strip()) > 0):
        ret += " (" + clean_name(location) + ")"
    return ret


def get_description(experiencerecord):
    desc_elt = experiencerecord.find('Description')
    if desc_elt is not None:
        if desc_elt.text is not None:
            return desc_elt.text
    return ""


def clean_name(s):
    if s is not None:
        try:
            clean = str(s)
        except UnicodeEncodeError:
            clean = s.encode('ascii', 'ignore')
        clean = clean.lower().translate(string.maketrans("", ""), string.punctuation)
        return clean
    else:
        return ""


def make_date(year_elt, month_elt):
    if month_elt is None:
        month_elt = 1
    else:
        month_elt = int(month_elt.text)
        if month_elt == 0:
            month_elt = 1

    if year_elt is None:
        return None
    else:
        year_elt = int(year_elt.text)
        try:
            return datetime.date(year_elt, month_elt, 1)
        except ValueError:
            # sys.stderr.write("cannot make date({}, {}, 1)\n".format(year_elt, month_elt))
            return None


def sort_stints(stints):
    # stints_filled = []
    # for start, end, desc in stints:
    #     if (start is None) and (end is None):
    #         # stints_filled.append((start, end, desc))
    #         pass
    #     elif start is None:
    #         stints_filled.append((end, end, desc))
    #     elif end is None:
    #         stints_filled.append((start, start, desc))
    #     else:
    #         stints_filled.append((start, end, desc))
    # return sorted(stints_filled)
    # sys.stderr.write("stints:      {}\n".format(stints))

    stints_sorted = sorted([ s for s in stints if ((s[0] is not None) or (s[1] is not None)) ],
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
            stints_filled.append((start, end, co, desc))
        # sys.stderr.write("stints fill: {}\n".format(stints))
        stints_sorted = stints_filled

    return stints_sorted


def parse_timeline(resume):
    stints = []

    experience = resume.find('experience')
    if experience is not None:
        for experiencerecord in experience:
            company_name = get_company_name(experiencerecord)
            start = make_date(experiencerecord.find('StartYear'), experiencerecord.find('StartMonth'))
            end = make_date(experiencerecord.find('EndYear'), experiencerecord.find('EndMonth'))
            desc = get_description(experiencerecord)

            stints.append((start, end, company_name, desc))

    # education = resume.find('education')
    # if education is not None:
    #     for schoolrecord in education:
    #         grad_date = make_date(schoolrecord.find('CompleteYear'), schoolrecord.find('CompleteMonth'))
    #         school_name = schoolrecord.find('School').text if schoolrecord.find('School').text is not None else ""
    #         school_id = schoolrecord.find('SchoolId').text if schoolrecord.find('SchoolId').text is not None else ""
    #         school = unidecode.unidecode("EDU " + school_id + " " + school_name)
    #         stints.append((None, grad_date, school))

    return sort_stints(stints)


#zzz todo: this is broken with new format that includes description text
def check_overlaps(timeline1, timeline2, exclude_dups=False):
    companies1 = set([ d for s, e, d in timeline1 ])
    companies2 = set([ d for s, e, d in timeline2 ])
    if exclude_dups and (companies1 == companies2):
        return None
    common = companies1.intersection(companies2)
    overlaps = []
    if len(common) > 0:
        for s1, e1, d1 in [ (s, e, d) for s, e, d in timeline1 if (d in common) ]:
            for s2, e2, d2 in [(s, e, d) for s, e, d in timeline2 if (d == d1)]:
                if (s2 < e1) and (s1 < e2):
                    overlaps.append(d1)
    return set(overlaps)


def timeline2pretty(idx, timeline):
    ret = "timeline {}:\n".format(idx)
    for s, e, c, d in timeline:
        ret += "\t{}  {} ({:.1f} yrs)     {:30}\n".format(s, e, (e - s).days / 365.25, c)
    return ret


def xml2timelines(infile_names):
    timelines = []

    for f, infile_name in enumerate(infile_names):
        sys.stderr.write("parsing xml {}/{} {}\n".format(f+1, len(infile_names), infile_name))

        if infile_name.endswith('.gz'):
            with gzip.open(infile_name, 'rb') as infile:
                tree = ET.parse(infile)
        else:
            tree = ET.parse(infile_name)

        root = tree.getroot()
        for i, resume in enumerate(root.findall('resume')):
            if i % 1000 == 0:
                sys.stderr.write("{}\n".format(i))
            timeline = parse_timeline(resume)
            # sys.stderr.write(timeline2pretty(resume.find('ResumeID').text, timeline) + "\n\n")
            if len(timeline) > 0:
                timelines.append(timeline)
    return timelines


def timelines2descs(timelines):
    descs = []
    for timeline in timelines:
        for start, end, company_name, desc in timeline:
            descs.append(desc.lower())
    return "\n".join(descs)


def find_bigrams(timelines, num=None, stops=None):
    if stops is None:
        stops = {}
    descs = timelines2descs(timelines)
    sys.stderr.write("got {} MB of desc\n".format(sys.getsizeof(descs)/float(1024*1024)))

    tokens = nltk.wordpunct_tokenize(descs)
    sys.stderr.write("found {} tokens\n".format(len(tokens)))

    finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    sys.stderr.write("got bigrams\n")

    scored = finder.score_ngrams(nltk.collocations.BigramAssocMeasures().likelihood_ratio)
    sys.stderr.write("scored {} bigrams\n".format(len(scored)))

    # only keep bigrams where both words are alpha and aren't stop words
    scored_filt = [ (b, s) for b, s in scored if ((b[0] not in stops and b[1] not in stops) and
                                                  (b[0].isalpha() and b[1].isalpha())) ]

    if num is not None:
        scored_filt = scored_filt[:num]
    scored_filt.sort(key=lambda x: x[1], reverse=True)

    for bigram, score in scored_filt:
        print bigram, "\t", score, "*" if bigram[0] in stops else "-",\
            "*" if bigram[1] in stops else "-"

    return sorted(scored_filt, key=lambda x: x[1], reverse=True)
    # return set([ b for b, s in scored_filt ])
    # return { bg:s for bg, s in scored_filt }


# do this a little funkily for performance reasons
def replace_bigrams(toks, bigram__score):
    matches = []
    for i in range(len(toks)-1):
        bg = (toks[i], toks[i + 1])
        if bg in bigram__score:
            matches.append(bg)
    toks_replaced = toks[:]
    for bg in sorted(matches, key=lambda x: bigram__score[x], reverse=True):
        toks_replaced = replace_bigram(toks_replaced, bg)
    return toks_replaced


def replace_bigram(toks, bg):
    toks_ret = []
    i = 0
    while i < len(toks)-1:
        if (toks[i], toks[i+1]) == bg:
            toks_ret.append(BIGRAM_DELIM.join(bg))
            i += 2
        else:
            toks_ret.append(toks[i])
            i += 1
    return toks_ret


def clean_timeline_descs(timelines):
    stops = set(stopwords.words('english'))
    bigrams = find_bigrams(timelines, num=1000, stops=stops)
    bigram_vals = { BIGRAM_DELIM.join(bg) for bg, s in bigrams }
    bigram__score = { bg:s for bg, s in bigrams }

    sys.stderr.write("cleaning %d timelines\n" % len(timelines))

    timelines_clean = []
    for i, timeline_dirty in enumerate(timelines):
        timeline_clean = []
        for start, end, company_name, desc in timeline_dirty:
            desc_tokens = [ w.lower() for w in nltk.wordpunct_tokenize(desc) ]
            desc_tokens_repl = replace_bigrams(desc_tokens, bigram__score)
            desc_words = [ w for w in desc_tokens_repl if (w.isalpha() or (w in bigram_vals)) and
                                                          (w not in stops) ]
            timeline_clean.append((start, end, company_name, desc_words))
        timelines_clean.append(timeline_clean)

        if (i % 1000) == 0:
            sys.stderr.write("%d\t%s\n" % (i, str(timeline_dirty)))
            sys.stderr.write("%d\t%s\n\n" % (i, str(timeline_clean)))

    return timelines_clean


#####################################
if __name__ == '__main__':
    USAGE = " usage: " + sys.argv[0] + " infile0.tgz [infile1.tgz infile2.tgz ...] outfile.p"
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    ins = sys.argv[1:-1]
    out = sys.argv[-1]

    sys.stderr.write(str(ins) + "\n")
    timelines = xml2timelines(ins)
    sys.stderr.write("read {} timelines\n".format(len(timelines)))

    timelines_clean = clean_timeline_descs(timelines)
    sys.stderr.write("cleaned {} timelines\n".format(len(timelines_clean)))

    with open(out, 'wb') as outp:
        pickle.dump(timelines_clean, outp)




    # # check for overlaps
    # timelines.sort(key=lambda x: x[0][0])
    # done = 0
    # for i in range(len(timelines)-1):
    #     for j in range(i+1, len(timelines)):
    #         if (i % 1000 == 0) and (j % 1000 == 0):
    #             tot = len(timelines)*len(timelines)
    #             done = i*len(timelines) + j
    #             # sys.stderr.write("done: {} / {}\n".format(done, tot))
    #             sys.stderr.write("\ti={}, j={}    {}%\n".format(i, j, int(float(done)/tot*100)))
    #             sys.stderr.flush()
    #         # s1 = timelines[i][0][0]
    #         e1 = timelines[i][-1][1]
    #         s2 = timelines[j][0][0]
    #         # e2 = timelines[j][-1][1]
    #         if s2 > e1:
    #             break
    #
    #         overlaps = check_overlaps(timelines[i], timelines[j], exclude_dups=True)
    #         if (overlaps is not None) and (len(overlaps) > 1):
    #             sys.stderr.write("\nOVERLAPS ({}): {}\n".format(len(overlaps), overlaps))
    #             sys.stderr.write(timeline2str(timelines[i]) + "\n")
    #             sys.stderr.write(timeline2str(timelines[j]) + "\n")
    #             sys.stderr.write("\n")
    #
    #             sys.stdout.write("\nOVERLAPS ({}): {}\n".format(len(overlaps), overlaps))
    #             sys.stdout.write(timeline2str(timelines[i]) + "\n")
    #             sys.stdout.write(timeline2str(timelines[j]) + "\n")
    #             sys.stdout.flush()
