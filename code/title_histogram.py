'''
I'm curious about the distribution of job titles in the resume dataset. Let's take a look.

@author: Dan Saunders (djsaunde.github.io)
'''

import cPickle as p


data = p.load(open('../data/sequential_title_data.p', 'rb'))
title_counts = {}

for title in set([ datum for l in data for datum in l ]):
	title_counts[title] = ([ datum for l in data for datum in l ]).count(title)

print title_counts.items()