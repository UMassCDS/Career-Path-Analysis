import sys
import collections
import xml.etree.ElementTree as ET
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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


#############################
if __name__ == '__main__':

    ROWS_PER_FIG = 3
    COLS_PER_FIG = 1

    infile_name = sys.argv[1]
    target_elt = sys.argv[2]

    sys.stderr.write("parsing xml\n")
    tree = ET.parse(infile_name)
    root = tree.getroot()

    sys.stderr.write("creating attr hists\n\n")
    attrs = get_attr_hists(root, target_elt)

    fig = plt.figure(figsize=(8, 10))

    for i, (attr_name, attr_counter) in enumerate(sorted(attrs.items())):
        sys.stderr.write("{}.{}\n".format(target_elt, attr_name))

        idx = (i % (ROWS_PER_FIG * COLS_PER_FIG)) + 1
        sys.stderr.write("\tplotting hist for {} (i={}, idx={})\n".format(attr_name, i, idx))

        # df = pd.DataFrame(sorted(attr_counter.items()))
        # df.columns = ["value", "count"]
        df = pd.DataFrame.from_records(sorted(attr_counter.items()), index="value", columns=["value", "count"])
        print df

        plot_bars(attr_name, df, idx)

        if (idx == ROWS_PER_FIG*COLS_PER_FIG) or (i == (len(attrs)-1)):
            save_plots(target_elt, i)
            fig = plt.figure(figsize=(8, 10))

        sys.stderr.write("\n\n")
    # save_plots(target_elt, len(attrs))


