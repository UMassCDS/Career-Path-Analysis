import sys
import collections
import xml.etree.ElementTree as ET
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def get_attr_hists(root, target_elt, topn=10, name_other="* other *", name_missing="* missing *"):
    attr_hists = {}  # attr_name -> value counter

    elts = root.findall(target_elt)
    for elt in elts:
        for attr_elt in elt:
            child_count = len(attr_elt)
            if child_count > 0:
                attr_name = attr_elt[0].tag + "_count"
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


# def add_margin(ax, x=0.05, y=0.05):
#     # This will, by default, add 5% to the x and y margins. You
#     # can customise this using the x and y arguments when you call it.
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     xmargin = (xlim[1]-xlim[0])*x
#     ymargin = (ylim[1]-ylim[0])*y
#     ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
#     ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)

class ExceptStupid(Exception):
    pass


# def plot_histogram(count_array, feature):
#     '''
#     Plot a histogram of the number of the feature per resume.
#     '''
#     plt.figure(figsize=(16, 9))
#     plt.hist(count_array, bins=range(max(count_array) + 1), rwidth=0.9)
#     plt.title('Histogram of ' + feature + ' per Resume')
#     plt.xlabel(feature + ' value')
#     plt.ylabel('Number of resumes')
#     plt.xticks(np.arange(0, max(count_array) + 1))
#
#     plt.savefig('../plots/%s_per_resume_histogram.png', feature)


pd.options.display.mpl_style = 'default'

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

    # ax.bar(edgecolor=None)

    try:
        # ax = sns.barplot(x="count", y=attr_name, data=df, label="zzz", color="b")

        ax = df.plot(y="count", x=attr_name, kind='barh', ax=ax, legend=False)

        sys.stderr.write("\tplotted {}\n".format(attr_name))
    except ExceptStupid:
        sys.stderr.write("\tunable to plot {}\n".format(attr_name))
        return

    # add_margin(ax, x=0.1, y=0.1)
    ax.set_title(attr_name)
    ax.set(xlabel="", ylabel="")


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

    for i, (attr_name, attr_counter) in enumerate(attrs.items()):
        sys.stderr.write("{}.{}\n".format(target_elt, attr_name))

        idx = (i % (ROWS_PER_FIG * COLS_PER_FIG)) + 1
        sys.stderr.write("\tplotting hist for {} (i={}, idx={})\n".format(attr_name, i, idx))

        df = pd.DataFrame(sorted(attr_counter.items()))
        df.columns = [attr_name, "count"]
        print df

        plot_bars(attr_name, df, idx)

        if idx == ROWS_PER_FIG*COLS_PER_FIG:
            save_plots(target_elt, i)
            fig = plt.figure(figsize=(8, 10))

        sys.stderr.write("\n\n")
    save_plots(target_elt, len(attrs))


