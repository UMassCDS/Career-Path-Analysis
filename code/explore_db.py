import argparse
import getpass
import random
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2 as db


def get_connection(host, dbname, user=None):
    if user is None:
        user = raw_input("db username: ")
    passw = getpass.getpass("db password: ")
    connstr = "host='{}' dbname={} user={} password='{}'".format(host, dbname, user, passw)
    return  db.connect(connstr)


def plot_distrib(conn, table_name, attr_name, num_samples=None, xlim=None, ylim=None, bw='scott'):
    # sql = "SELECT " + attr_name + " FROM " + table_name + " WHERE " + attr_name + " IS NOT NULL"
    sql = "SELECT " + attr_name + " FROM " + table_name + " WHERE " + attr_name + " < 10000"
    # if num_samples:
    #     sql += " LIMIT {}".format(num_samples)
    curs = conn.cursor()
    print sql
    curs.execute(sql)
    vals = [row[0] for row in curs.fetchall()]
    print "got {} vals".format(len(vals))
    if num_samples:
        random.shuffle(vals)
        vals = vals[:num_samples]
        print "kept {} vals".format(len(vals))
    # ax = sns.distplot(vals, kde=True, hist=False, kde_kws={'bw': bw})
    ax = sns.distplot(vals, kde=False, hist=True, bins=10000)
    # ax = sns.distplot(vals, kde=True, hist=False)
    # ax.set_xscale('linear')
    # ax.set_xlim()
    plt.yscale('linear')
    plt.xscale('linear')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    print plt.xlim(), plt.ylim()
    plt.show()
    print plt.xlim(), plt.ylim()


##################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import resume data into relational db')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default=None)
    parser.add_argument('--db', default=None)
    args = parser.parse_args()

    conn = get_connection(args.host, args.db, args.user)


