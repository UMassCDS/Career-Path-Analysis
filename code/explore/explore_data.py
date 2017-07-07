import sys
import collections
import xml.etree.ElementTree as ET
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import string
import unidecode

LABEL_OTHER = "* other *"
LABEL_MISSING = "* missing *"


def get_attr_hists(root, target_elt, topn=10, name_other=LABEL_OTHER, name_missing=LABEL_MISSING):
    attr_hists = {}  # attr_name -> value counter

    elts = root.findall(target_elt)
    for elt in elts:
        for attr_elt in elt:
            child_count = len(attr_elt)
            if child_count > 0:
                attr_name = attr_elt.tag + "." + attr_elt[0].tag + "_count"
                attr_val = child_count
            else:
                attr_name = attr_elt.tag
                attr_val = attr_elt.text

            if attr_name not in attr_hists:
                attr_hists[attr_name] = collections.Counter()
            attr_hists[attr_name][attr_val] += 1

    attr_hists_sm = {}
    for attr_name, attr_hist in attr_hists.items():
        attr_hist_sm = counter_resize(attr_hist, topn, name_other)
        val_count = sum(attr_hist_sm.values())
        if val_count < len(elts):
            attr_hist_sm[name_missing] += (len(elts) - val_count)
        attr_hists_sm[attr_name] = attr_hist_sm
    return attr_hists_sm


def counter_resize(cnt, n, other_name):
    tops = collections.Counter()
    for v, c in cnt.most_common(n-1):
        tops[v] = c
    size_all = sum(cnt.values())
    size_tops = sum(tops.values())
    size_other = size_all - size_tops
    if size_other:
        tops[other_name] = size_other
    return tops


# pd.options.display.mpl_style = 'default'
# plt.style.use("ggplot")
# plt.
import matplotlib
# matplotlib.style.use('ggplot')
# matplotlib.pyplot.use('ggplot')
matplotlib.pyplot.style.use('ggplot')

def plot_bars(attr_name, df, idx):
    # sns.set(style="whitegrid")
    # sns.set_color_codes("pastel")

    # plt.tight_layout()
    ax = plt.subplot(ROWS_PER_FIG, COLS_PER_FIG, idx)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # left = 0.125  # the left side of the subplots of the figure
    # right = 0.9  # the right side of the subplots of the figure
    # bottom = 0.1  # the bottom of the subplots of the figure
    # top = 0.9  # the top of the subplots of the figure
    # wspace = 0.2  # the amount of width reserved for blank space between subplots
    # hspace = 0.2  # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=0.2, right=0.8, hspace=1.0, wspace=1.0)


    xlim = None
    # df.sort_values("count", inplace=True)
    # if df.iloc[-1][attr_name] == LABEL_OTHER:
    #     xlim = (0, 1.05*df.iloc[-2]["count"])

    val_other = df.get_value(LABEL_OTHER, "count") if (LABEL_OTHER in df.index) else 0
    val_missing = df.get_value(LABEL_MISSING, "count") if (LABEL_MISSING in df.index) else 0


    # val_other = df[]
    # val_other = df.loc[df[attr_name] == LABEL_OTHER, "count"]
    # val_other = df.get_value(LABEL_OTHER, "count")
    # val_missing = df.loc[df[attr_name] == LABEL_MISSING]["count"][0]
    print "\n\n***other: {}, missing: {}***".format(val_other, val_missing)

    # df_data = df[(df["value"] != LABEL_OTHER) & (df["value"] != LABEL_MISSING)]
    df_data = df.loc[(df.index != LABEL_MISSING) & (df.index != LABEL_OTHER)]
    print df_data


    df_data.sort_values("count", inplace=True)
    val_top = df_data.iloc[-1]["count"]
    print "\n\n***top: {}***".format(val_top)

    if (val_missing > 2*val_top) or (val_other > 2*val_top):
        xlim = (0, 1.05*val_top)
        print "\n\n***xlim: {}***".format(xlim)


    # print "second last one: ", df.iloc(-2)
    # print "last one: ", df.iloc(-1)

    ax = df.plot(y="count", x=None, kind='barh', ax=ax, legend=False, xlim=xlim)
    ax.set_title(attr_name)
    ax.set(xlabel="", ylabel="")

    for rect, val in zip(ax.patches, df["count"]):
        # h = rect.get_height()
        # ax.text(rect.get_x() + rect.get_width()/2, h + 5, "stupid", ha='center', va='bottom')
        h = rect.get_height()
        w = rect.get_width()
        sty = 'normal'

        x = rect.get_x() + w
        if (xlim is not None) and (val > xlim[1]):
            # val = "! " + str(val)
            x = xlim[1]
            sty = 'italic'
        ax.text(x, rect.get_y() + h/2, val, ha='left', va='center', size='x-small', style=sty)


def save_plots(elt, index):
    plt.savefig("{}_attrs_{}.png".format(elt.replace('/', '-'), index))


def overlap(start_year1, end_year1, start_year2, end_year2):
    if start_year1 is None:
        if end_year1 is None:
            return False
        else:
            start_year1 = end_year1

    if start_year2 is None:
        if end_year2 is None:
            return False
        else:
            start_year2 = end_year2

    if start_year1 <= start_year2:
        return start_year2 <= end_year1
    else:  # (start_year1 > start_year2)
        return start_year1 < end_year2


#############################
if __name__ == '__main__':

    ROWS_PER_FIG = 3
    COLS_PER_FIG = 1

    infile_names = sys.argv[1:-1]
    target_elt = sys.argv[-1]

    print sys.argv
    print infile_names
    print target_elt

    # if 0:
    #     sys.stderr.write("creating attr hists\n\n")
    #     attrs = get_attr_hists(root, target_elt)
    #
    #     fig = plt.figure(figsize=(8, 10))
    #
    #     for i, (attr_name, attr_counter) in enumerate(sorted(attrs.items())):
    #         sys.stderr.write("{}.{}\n".format(target_elt, attr_name))
    #
    #         idx = (i % (ROWS_PER_FIG * COLS_PER_FIG)) + 1
    #         sys.stderr.write("\tplotting hist for {} (i={}, idx={})\n".format(attr_name, i, idx))
    #
    #         # df = pd.DataFrame(sorted(attr_counter.items()))
    #         # df.columns = ["value", "count"]
    #         df = pd.DataFrame.from_records(sorted(attr_counter.items()), index="value", columns=["value", "count"])
    #         print df
    #
    #         plot_bars(attr_name, df, idx)
    #
    #         if (idx == ROWS_PER_FIG*COLS_PER_FIG) or (i == (len(attrs)-1)):
    #             save_plots(target_elt, i)
    #             fig = plt.figure(figsize=(8, 10))
    #
    #         sys.stderr.write("\n\n")
    #     # save_plots(target_elt, len(attrs))

    if 0:
        company__persons = {}

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
                resume_id = resume.find('ResumeID').text

                experience = resume.find('experience')
                if experience is not None:
                    for experiencerecord in experience:
                        company_name = experiencerecord.find('Company').text
                        location = experiencerecord.find('Location').text if experiencerecord.find('Location') else None

                        try:
                            company_name = str(company_name)
                        except UnicodeEncodeError:
                            company_name = str(unidecode.unidecode(company_name))
                        company_name = company_name.lower().translate(string.maketrans("",""), string.punctuation)

                        if location is not None:
                            company_name += " " + unidecode.unidecode(location)

                        # start_month = experiencerecord.find('StartMonth')
                        start_year = experiencerecord.find('StartYear')
                        if start_year is not None:
                            start_year = int(start_year.text)
                        # end_month = experiencerecord.find('EndMonth')

                        end_year = experiencerecord.find('EndYear')
                        if end_year is not None:
                            end_year = int(end_year.text)

                        company__persons.setdefault(company_name, []).append((resume_id, start_year, end_year))
                        # company__persons.setdefault(company_name, []).append(resume_id)

        links = []
        for company, persons in company__persons.items():
            for i in range(len(persons) - 1):
                for j in range(i+1, len(persons)):
                    id1, s1, e1 = persons[i]
                    id2, s2, e2 = persons[j]
                    if overlap(s1, e1, s2, e2):
                        sys.stderr.write("link {}: {}-{}, {}-{}\n".format(company, s1, e1, s2, e2))
                        links.append((id1, id2, company))

        person_link_count = collections.Counter()
        for id1, id2, company in links:
            person_link_count[id1] += 1
            person_link_count[id2] += 1
        for pid, cnt in sorted(person_link_count.items(), key=lambda x: x[1], reverse=True)[:50]:
            sys.stderr.write("person {}, deg {}\n".format(pid, cnt))

        deg_company_tups = [(len(v), k) for k, v in company__persons.items()]
        deg_company_tups.sort(reverse=True)

        for deg, company in deg_company_tups[:50]:
            sys.stderr.write("{} {}\n".format(company, deg))


    if 1:

        infile_name = sys.argv[1]

        def print_kids(elt, elt__kids, indent=0):
            delimit = '/'
            if delimit in elt:
                prefix, tag = elt.rsplit("/", 1)
                print " "*indent, tag
            else:
                print " "*indent, elt

            if elt in elt__kids:
                for kid in sorted(elt__kids.get(elt, [])):
                    print_kids(kid, elt__kids, indent + 4)

        infile_name = sys.argv[1]
        sys.stderr.write("parsing xml {}\n".format(infile_name))

        elt__children = {}
        schema_todo = {'resume'}
        schema_done = set()

        if infile_name.endswith('.gz'):
            with gzip.open(infile_name, 'rb') as infile:
                tree = ET.parse(infile)
        else:
            tree = ET.parse(infile_name)
        root = tree.getroot()

        while len(schema_todo) > 0:

            elt_tag = schema_todo.pop()
            elt__children[elt_tag] = set()
            sys.stderr.write("exploring {}\n".format(elt_tag))

            for i, elt in enumerate(root.findall(elt_tag)):
                for elt_child in elt:
                    elt_child_tag = elt_tag + '/' + elt_child.tag

                    if (elt_child_tag not in schema_todo) and (elt_child not in schema_done):
                        schema_todo.add(elt_child_tag)
                    if elt_child_tag not in elt__children[elt_tag]:
                        sys.stderr.write("\t found {}\n".format(elt_child_tag))
                        elt__children[elt_tag].add(elt_child_tag)

            schema_done.add(elt_tag)


        print_kids("resume", elt__children)







