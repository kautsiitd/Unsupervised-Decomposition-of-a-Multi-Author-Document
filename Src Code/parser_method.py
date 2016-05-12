# -*- coding: utf-8 -*-
# encoding=utf8
from __future__ import unicode_literals
'''#############################'''
'''#####Importing Libraries#####'''
###################################
import sys
import os
import io
import re
import pickle
from random import randint as rnd
from random import shuffle
from itertools import groupby
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.mixture import GMM
from pprint import pprint
import operator
import math
import nltk
from sklearn.naive_bayes import MultinomialNB as BNB
import itertools
import numpy as np
from multiprocessing import Pool
from multiprocessing import Manager
# from pos_tag import pos_tagging
# from pq_gram import find_pq_grams
print "Import Done"
'''#############################'''
'''#####Importing Libraries#####'''
###################################

'''############################'''
'''#####Defining Variables#####'''
##################################
V 		  = 50
b_num     = 0
b = ["Becker-Posner","GC-TF-PK","MD-TF-PK","MD-GC-PK","MD-GC-TF-PK"]
gmm_initialisation = 5
	# segment size
seg_size  = 30
	# number of sentence to test on from final model
n_gram_size = 3
'''############################'''
'''#####Defining Variables#####'''
##################################

'''##########Step 1###########'''
'''extracting and merging data'''
#################################
'''
	books_names = ['a',b','c',....]
	merged_data = ['sentence1','sentence2',.....]
	label_sen   = [0,0,0,1,2,.....]								label of sentence in merged_data
	segments_sen= [['sentence1','sentence2',... ],['sentence1','sentence2',... ],... number of segments]
	segments    = ['segment1','segment2',.....]
	label_in_seg= [[0,0,0,1,2,0,0,..]
				   [0,1,1,2,1,0,1,..]
				    ....
				  ]	label of sentences in individual segments
	label_in_seg= [0,1,1,0,2,0,...]								book with max count in segment
	is_pure_seg = [True,True,False,True,... number of segments]	True if segment is pure
	randoms		= [ [120,[2,0,1,.. number of Authors]], [180,[1,2,0,.. number of Authors]], .. ]
'''
folder 		= "../dataset/Original/"+b[b_num]
books_names = os.listdir(folder)
merged_data	= []
label_sen	= []
segments_sen= []
segments 	= []
label_seg   = []
label_in_seg= []
is_pure_seg = []
randoms		= []
	# main
number_books= len(books_names)
books_data 	= []
for book in books_names:
	path = os.path.join(folder,book)
	f    = io.open(path, encoding="ISO-8859-1")
	books_data.append(f.readlines())
number_sen	= [len(book_data) for book_data in books_data]
total_sen	= sum(number_sen)
number_seg	= int(math.ceil((total_sen/seg_size)))
count_sen 	= [0]*number_books
while(sum(count_sen) != total_sen):
	size		  = rnd(1,V)
	randoms.append([size,[]])
	done_book 	  = [0]*number_books
	for i in range(number_books):
		book_num	  = rnd(0,number_books-1)
		while(done_book[book_num] != 0):
			book_num	  = rnd(0,number_books-1)
		randoms[-1][-1].append(book_num)
		done_book[book_num] = 1
		new_count_sen = count_sen[book_num] + min(size,number_sen[book_num]-count_sen[book_num])
		for j in books_data[book_num][ count_sen[book_num]:new_count_sen ]:
			merged_data.append( re.sub('[\r\n]','',j) )
		label_sen.extend([book_num] * (new_count_sen - count_sen[book_num]) )
		count_sen[book_num]	= new_count_sen
for i in range(number_seg):
	start = seg_size*i
	end = min(seg_size*(i+1),total_sen)
	seg_data = merged_data[start:end]
	segments_sen.append(seg_data)
	segments.append(' '.join(seg_data))
	labels = label_sen[start:end]
	label_in_seg.append(labels)
for i in range(number_seg):
	label_seg.append(max(set(label_in_seg[i]), key=label_in_seg[i].count))
for i in range(number_seg):
	is_pure_seg.append(sum(label_in_seg[i])%len(label_in_seg[i]) == 0)
'''######'''
'''Step 1'''
'''######'''

'''###########################'''
'''Printing Results of merging'''
'''###########################'''
'''
	org_seg = [430,405,...,150]							number of pure segments by author i, last one for mixed
'''
	# calculating segments by each author
org_seg		= [0 for i in range(number_books+1)]
for i in range(number_seg):
	if( sum(label_in_seg[i])%len(label_in_seg[i]) == 0):
		org_seg[ sum(label_in_seg[i])/len(label_in_seg[i]) ] += 1
	else:
		org_seg[-1] += 1
for i in range(number_books):
	print "Author "+str(i)+":",org_seg[i]
print "Mixed   :",org_seg[-1]
print "STEP 1 done"
'''###########################'''
'''Printing Results of merging'''
#################################

'''##########Step 2##########'''
'''Get pq-gram of merged data'''
################################
'''
	segments_parser = [ [[**ROOTNN*,*NN**JJ, ... pq-grams of Sentence 1],[**ROOTNN*,*NN**JJ, ... pq-grams of Sentence 2],....
																						... Number of Sentence in Segment 1]
						[[**ROOTNN*,*NN**JJ, ... pq-grams of Sentence 1],[**ROOTNN*,*NN**JJ, ... pq-grams of Sentence 2],....
																						... Number of Sentence in Segment 2]
						....
						Number of segments
					  ]
'''
folder 		= "../dataset/Parser/"+b[b_num]
books_names = os.listdir(folder)
merged_parser = []
segments_parser = []
parser_data 	= []
count_sen 	= [0]*number_books
for book in books_names:
	path = os.path.join(folder,book)
	f    = io.open(path, encoding="ISO-8859-1")
	parser_data.append(f.readlines())
for random in randoms:
	size		  = random[0]
	for i in range(number_books):
		book_num	  = random[-1][i]
		new_count_sen = count_sen[book_num] + min(size,number_sen[book_num]-count_sen[book_num])
		for j in parser_data[book_num][ count_sen[book_num]:new_count_sen ]:
			merged_parser.append( re.sub('[\r\n]','',j) )
		count_sen[book_num]	= new_count_sen
for i in range(number_seg):
	start = seg_size*i
	end = min(seg_size*(i+1),total_sen)
	seg_data = merged_parser[start:end]
	segments_parser.append(seg_data)

'''#########################STEP3#######################'''
'''Find If given Segment is Pure or Mix using words used'''
###########################################################
'''
Calculate Similiarity Index of Segment:
	Higher Similiarity Index, Higher Pure
'''

# sys.exit()

'''################STEP4###############'''
'''Find If given Segment is Pure or Mix'''
##########################################
'''
Calculate Similiarity Index of Segment:
	Higher Similiarity Index, Higher Pure
'''
import matplotlib.pyplot as plt
from scipy.interpolate import spline
sentence_size  = []
# calculating sentence sizes in each segment
for segment in segments_parser:
	sentence_size.append([])
	for sentence in segment:
		sentence_size[-1].append(len(sentence))

# calculating similiarity index for each segment
number_seg = len(segments_parser)
score_true = Manager().list([])
score_false= Manager().list([])
def score_similiarity(i):
	segment = segments_parser[i]
	similiarity_index = 0
	seg_size = len(segment)
	'''iterating over sentences in a segment'''
	for j in range(seg_size):
		for pq_gram in segment[j]:
			'''checking current pqgram is in how 
			many other sentences of same segment'''
			for k in range(j,seg_size):
				if pq_gram in segment[k]:
					similiarity_index += 1.0/(sentence_size[i][j]*sentence_size[i][k])
	# for j in range(len(label_in_seg[i])):
	# 	print segments_sen[i][j].encode("ISO-8859-1"),label_in_seg[i][j]
	# print is_pure_seg[i],similiarity_index
	# sys.exit()
	if is_pure_seg[i] == True:
		score_true.append(similiarity_index)
	else:
		score_false.append(similiarity_index)
p = Pool(8)
p.map(score_similiarity, range(number_seg))
print "Accuracy Initial",float(sum(org_seg)-org_seg[-1])/sum(org_seg),sum(org_seg)
accuracies = []
n_pure = []
data_size = []
fig, ax = plt.subplots()
axes = [ax, ax.twinx()]
for thr in range(20,210):
	mixed_segments = []
	pure_segments  = []
	threshold = float(thr)/100
	for similiarity_index in score_true:
		if similiarity_index > threshold:
			pure_segments.append(1)
		else:
			mixed_segments.append(0)
	for similiarity_index in score_false:
		if similiarity_index > threshold:
			pure_segments.append(0)
		else:
			mixed_segments.append(1)
	print thr,"Accuracy Final",pure_segments.count(1),float(pure_segments.count(1))/len(pure_segments),len(pure_segments)
	accuracies.append(float(pure_segments.count(1)*100)/len(pure_segments))
	n_pure.append(pure_segments.count(1))
	data_size.append(len(pure_segments))

base = np.array([float(x)/100 for x in range(20,210)])
thr = np.linspace(base.min(),base.max(),500)
accuracies_smooth = spline(base,accuracies,thr)
n_pure_smooth = spline(base,n_pure,thr)
data_size_smooth = spline(base,data_size,thr)
axes[1].plot(thr,accuracies_smooth,'r')
axes[0].plot(thr,n_pure_smooth,'b')
axes[0].plot(thr,data_size_smooth,'g')
plt.show()
	# score_true.sort()
	# score_false.sort()
	# print score_true
	# print score_false