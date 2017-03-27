import sys
import collections
import xml.etree.ElementTree as ET
import seaborn as sns
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


def add_margin(ax, x=0.05, y=0.05):
    # This will, by default, add 5% to the x and y margins. You
    # can customise this using the x and y arguments when you call it.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y
    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)


def plot_bars(df, colx, coly):
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    # f, ax = plt.subplots(figsize=(6, 6))

    sns.set_color_codes("pastel")
    sns.barplot(x=colx, y=coly, data=df, label="zzz", color="b")

    # Add a legend and informative axis label
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    # ax.set(xlim=(0, 24), ylabel="",
    #        xlabel="Automobile collisions per billion miles")

    # ax.set(xlabel=colx, ylabel=coly)

    # sns.despine(left=True, bottom=True)
    plt.savefig("bars_{}.png".format(coly))

#############################
if __name__ == '__main__':

    infile_name = sys.argv[1]
    target_elt = sys.argv[2]

    sys.stderr.write("parsing xml\n")
    tree = ET.parse(infile_name)
    root = tree.getroot()

    sys.stderr.write("creating attr hists\n")
    attrs = get_attr_hists(root, target_elt)

    sns.set(style="whitegrid")
    sns.set_color_codes("pastel")
    # f, axs = plt.subplots(nrows=20, ncols=2)

    f = plt.figure(figsize=(8, 10))

    for i, (attr_name, attr_counter) in enumerate(attrs.items()):
        sys.stderr.write("plotting hist for {}\n".format(attr_name))

        try:
            for v, c in sorted(attr_counter.items()):
                sys.stderr.write("\t{}:\t{}\n".format(v, c))
        except Exception:
            pass

        # df = pd.DataFrame.from_dict(attr_counter_top10, orient='columns', dtype=None)
        # print df
        # df.columns = ["count"]
        # df.index.names = [attr_name]
        # df.columns = [attr_name, "count"]
        # print df

        df = pd.DataFrame(sorted(attr_counter.items()))
        print df

        df.columns = [attr_name, "count"]
        print df

        # plot_bars(df, "count", attr_name)

        # 0     1
        # 1     2
        # 3     3
        # 4     1
        # 5     2
        # 6     3
        rows = 3
        cols = 1
        idx = (i % (rows*cols)) + 1

        # plots_per_page = rows*cols
        # j = i + 1
        # idx = (rows*cols % j) + 1
        sys.stderr.write("*** plot {}, index {}\n".format(i, idx))

        if (idx == 1) and (i > 0):
            plt.savefig("{}_attrs_{}.png".format(target_elt.replace('/', '-'), i - 1))
            sys.stderr.write("*** new fig\n")
            f = plt.figure(figsize=(8, 10))



        # ax = f.add_subplot(rows, cols, idx)

        s = plt.subplot(rows, cols, idx)

        try:
            ax = sns.barplot(x="count", y=attr_name, data=df, label="zzz", color="b")
        except Exception:
            continue

        add_margin(ax, x=0.1, y=0.1)


        ax.set_title(attr_name)
        ax.set(xlabel="", ylabel="")
        # axs[i].set(xlabel="count", ylabel=attr_name)
        # axs[i].set_title("hell")


        # sns.despine(left=True, bottom=True)
        # plt.savefig("bars_{}_{}.png".format(i, attr_name))

    plt.savefig("{}_attrs_{}.png".format(target_elt.replace('/', '-'), i - 1))



