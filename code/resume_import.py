import sys
import string
import datetime
import gzip
import json
import xml.etree.ElementTree as ET

import nltk
import nltk.collocations
from nltk.corpus import stopwords

import resume_common


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
        if year_elt < 100:
            year_elt += 1900
        try:
            return datetime.date(year_elt, month_elt, 1)
        except ValueError:
            # sys.stderr.write("cannot make date({}, {}, 1)\n".format(year_elt, month_elt))
            return None


def parse_resume(resume_xml):
    resume = []

    experience = resume_xml.find('experience')
    if experience is not None:
        for experiencerecord in experience:
            company_name = get_company_name(experiencerecord)
            start = make_date(experiencerecord.find('StartYear'), experiencerecord.find('StartMonth'))
            end = make_date(experiencerecord.find('EndYear'), experiencerecord.find('EndMonth'))
            desc = get_description(experiencerecord)

            resume.append(resume_common.ResumeEntry(start, end, company_name, desc))

    # education = resume.find('education')
    # if education is not None:
    #     for schoolrecord in education:
    #         grad_date = make_date(schoolrecord.find('CompleteYear'), schoolrecord.find('CompleteMonth'))
    #         school_name = schoolrecord.find('School').text if schoolrecord.find('School').text is not None else ""
    #         school_id = schoolrecord.find('SchoolId').text if schoolrecord.find('SchoolId').text is not None else ""
    #         school = unidecode.unidecode("EDU " + school_id + " " + school_name)
    #         stints.append((None, grad_date, school))

    return resume_common.sort_stints(resume)


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


def xml2resumes(infile_names):
    timelines = []

    for f, infile_name in enumerate(infile_names):
        sys.stderr.write("parsing xml {}/{} {}\n".format(f+1, len(infile_names), infile_name))

        if infile_name.endswith('.gz'):
            with gzip.open(infile_name, 'rb') as infile:
                tree = ET.parse(infile)
        else:
            tree = ET.parse(infile_name)

        root = tree.getroot()
        for i, resume_xml in enumerate(root.findall('resume')):
            if i % 1000 == 0:
                sys.stderr.write("{}\n".format(i))
            timeline = parse_resume(resume_xml)
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


def clean_resume_descs(resumes):
    stops = set(stopwords.words('english'))
    bigrams = find_bigrams(resumes, num=1000, stops=stops)
    bigram_vals = { BIGRAM_DELIM.join(bg) for bg, s in bigrams }
    bigram__score = { bg:s for bg, s in bigrams }

    sys.stderr.write("cleaning %d resumes\n" % len(resumes))
    resumes_clean = []
    for i, resume_dirty in enumerate(resumes):
        resume_clean = []
        # for start, end, company_name, desc in resume_dirty:
        #     desc_tokens = [ w.lower() for w in nltk.wordpunct_tokenize(desc) ]
        #     desc_tokens_repl = replace_bigrams(desc_tokens, bigram__score)
        #     desc_words = [ w for w in desc_tokens_repl if (w.isalpha() or (w in bigram_vals)) and
        #                                                   (w not in stops) ]

        # indexing by field name below might be slow, but it eases structural dependency with
        # resume_common.py
        for res_ent in resume_dirty:
            desc_tokens = [ w.lower() for w in nltk.wordpunct_tokenize(res_ent.desc) ]
            desc_tokens_repl = replace_bigrams(desc_tokens, bigram__score)
            desc_words = [ w for w in desc_tokens_repl if (w.isalpha() or (w in bigram_vals)) and
                                                          (w not in stops) ]
            desc_words_str = " ".join(desc_words)
            resume_clean.append(resume_common.ResumeEntry(res_ent.start,
                                                          res_ent.end,
                                                          res_ent.company,
                                                          desc_words_str))
        resumes_clean.append(resume_clean)

        if (i % 1000) == 0:
            sys.stderr.write("%d\t%s\n" % (i, str(resume_dirty)))
            sys.stderr.write("%d\t%s\n\n" % (i, str(resume_clean)))

    return resumes_clean


# Even though we only use dump() here, define them together so they stay in sync
def dump_json_resumes(resumes, outfile_name):
    with open(outfile_name, 'w') as outfile:
        json.dump([ [ resume_common.tuplify(r) for r in resume ] for resume in resumes ], outfile)


def load_json_resumes(infile_name):
    with open(infile_name, 'r') as infile:
        resume_tups = json.load(infile)
    return [ [ resume_common.detuplify(r) for r in resume_tup ] for resume_tup in resume_tups ]


#####################################
if __name__ == '__main__':
    USAGE = " usage: " + sys.argv[0] + " infile0.tgz [infile1.tgz infile2.tgz ...] outfile.json"
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    ins = sys.argv[1:-1]
    out = sys.argv[-1]

    sys.stderr.write(str(ins) + "\n")
    resumes = xml2resumes(ins)
    sys.stderr.write("read {} resumes\n".format(len(resumes)))

    resumes_clean = clean_resume_descs(resumes)
    sys.stderr.write("cleaned {} resumes\n".format(len(resumes_clean)))

    # with open(out, 'wb') as outp:
    #     pickle.dump(resumes_clean, outp)
    dump_json_resumes(resumes_clean, out)


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
