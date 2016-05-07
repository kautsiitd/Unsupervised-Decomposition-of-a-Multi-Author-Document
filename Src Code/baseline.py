# -*- coding: utf-8 -*-
# encoding=utf8
from __future__ import unicode_literals
'''#############################'''
'''#####Importing Libraries#####'''
'''#############################'''
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
# from pos_tag import pos_tagging
# from pq_gram import find_pq_grams
print "Import Done"
'''#############################'''
'''#####Importing Libraries#####'''
'''#############################'''

# variables
V 		  = 200
b_num     = 0
b = ["Becker-Posner","GC-TF-PK","MD-TF-PK","MD-GC-PK","MD-GC-TF-PK"]
gmm_initialisation = 5
	# number of most frequent features
max_features = 1500
	# segment size
seg_size  = 30
	# choosing top vital segment in a class
best_per  = .8
	# number of sentence to test on from final model
test_size = 1000
n_gram_size = 3

'''###########################'''
'''##########Step 1###########'''
'''extracting and merging data'''
'''###########################'''
'''
books_names = ['a',b','c',....]
merged_data = ['sentence1','sentence2',.....]
label_sen   = [0,0,0,1,2,.....]								label of sentence in merged_data
segments    = ['segment1','segment2',.....]
label_in_seg= [[0,0,0,1,2,0,0,..]
			   [0,1,1,2,1,0,1,..]
			    ....
			  ]	label of sentences in individual segments
label_in_seg   = [0,1,1,0,2,0,...]								book with max count in segment
'''
# STEP 1
# extracting and merging data
	# variables
	# Eze-Job
	# NewYorkTimesArticles
	# MD-TF
	# GC-MD
	# Becker-Posner
	# GC-TF-PK
	# MD-TF-PK
	# MD-GC-PK
	# MD-GC-TF-PK
folder 		= "dataset/Original/"+b[b_num]
books_names = os.listdir(folder)
merged_data	= []
label_sen	= []
segments 	= []
label_seg   = []
label_in_seg= []
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
	done_book 	  = [0]*number_books
	for i in range(number_books):
		book_num	  = rnd(0,number_books-1)
		while(done_book[book_num] != 0):
			book_num	  = rnd(0,number_books-1)
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
	segments.append(' '.join(seg_data))
	labels = label_sen[start:end]
	label_in_seg.append(labels)
for i in range(number_seg):
	label_seg.append(max(set(label_in_seg[i]), key=label_in_seg[i].count))
'''######'''
'''Step 1'''
'''######'''

'''###########################'''
'''Printing Results of merging'''
'''###########################'''
'''
org_seg = [430,405,...,150]									number of pure segments by author i, last one for mixed
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
'''###########################'''

'''#########################################'''
'''#################Step 2##################'''
'''finding features and vectorising segments'''
'''#########################################'''
'''
model = model with feature words having atleast frequency = 3 = 11000
vec_seg(sparse matrix) = [ [0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,... number of feature words=11000]
						 [0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,... whether word present or not]
						 ....
						 number of segments
		  				 ]
number_f_w = number of feature words extracted from merged data
'''
model		  = CV(binary = True, min_df = 3, ngram_range=(1,n_gram_size), max_features=20000)
model 		  = model.fit(merged_data)
vec_seg		  = model.transform(segments)
number_f_w	  = len(model.vocabulary_)
max_features  = min(max_features,number_f_w)
print "number of feature words:",number_f_w
print "STEP 2 done"
'''######'''
'''Step 2'''
'''######'''

'''############################################'''
'''#################Step 3#####################'''
'''Unsupervised labelling of segments using GMM'''
'''############################################'''
'''
label_p = [0,1,0,1,2,0,1,.... number of segments] 				predicted label for each segment
count_mapping = [[20,3,450,... number of books]					how much predicted label match to original label(max count)
				 [410,5,10,..]
				 ...
				 number of books 
				]
mapping 	  = [2,0,1,5,3,...]									What predicted label match to in original label
clusters	  = [['sentence','sentence',..... in cluster 0]
				 ['sentence','sentence',..... in cluster 1]
				 ....
				 number of books
				]
'''
mapping = [0 for i in range(number_books)]
while(len(set(mapping)) != number_books):
	model1		  = GMM(n_components = number_books, n_iter = 1000, covariance_type = 'diag', n_init = gmm_initialisation, verbose = 1)
	# label_p1 	  = model1.fit_predict(vec_seg.toarray())
	model1		  = model1.fit(vec_seg.toarray())
	label_p 	  = model1.predict_proba(vec_seg.toarray())
	temp_label_p  = []
	for j in range(number_seg):
		temp_label_p.append(map(lambda x: (x),label_p[j]) . index(max(label_p[j])))
	label_p = temp_label_p
	count_mapping = [ [0 for j in range(number_books)] for i in range(number_books)]
	for i,j in zip(label_p,label_seg):
		count_mapping[i][j] += 1
	for i in range(number_books):
		max_frq = max(count_mapping[i])
		mapping[i] = count_mapping[i].index(max_frq)
	print "mapping:",mapping
	print "count_mapping:",count_mapping

	# updating label_p with mapping
for i in range(number_seg):
	label_p[i] = mapping[label_p[i]]
	# segments in each clusters as sentences
clusters = [[] for i in range(number_books)]
for i in range(number_seg):
	clusters[label_p[i]].append(segments[i])
'''######'''
'''Step 3'''
'''######'''

'''################################'''
'''Calculating Precision and Recall'''
'''################################'''
confusion_matrix = [ [0 for j in range(number_books)] for i in range(number_books)]
for i in range(number_seg):
	confusion_matrix[label_p[i]][label_seg[i]] += 1
recall = []
for i in range(number_books):
	recall.append(float(confusion_matrix[i][i])/sum(zip(*confusion_matrix)[i]))
print "Recall:",recall
print float(sum(recall))/number_books
print "mapping:",mapping
print "confusion_matrix:",confusion_matrix
print "STEP 3 done"
'''################################'''
'''Calculating Precision and Recall'''
'''################################'''

'''############################################################'''
'''######################Step 4################################'''
'''Revectorising segments with max_features most frequent words'''
'''############################################################'''
'''
model2 = model with at most max_features=1500 feature words
vec_seg_cls(sparse matrix) = [[ [0,1,1,0,1,1,1,0,..... max_features=1500],[vector of segment 2],.... cluster 0]
			   				 [ [0,0,1,1,0,0,1,0,..... max_features=1500],[vector of segment 2],.... cluster 1]
			   				 ....
			   				 number of books
			  				 ]								vector representation of each segment in corresponding cluster
vec_seg_new(sparse matrix) = [[0,1,1,0,1,1,1,0,..... max_features=1500]
							  [0,0,1,1,0,0,1,0,..... max_features=1500]
							  ....
							  number of segments
							 ]								vector representation of each segment
'''
model2		  = CV(ngram_range=(1,n_gram_size), max_features = max_features)
model2 		  = model2.fit(merged_data)
vec_seg_cls   = [model2.transform(clusters[i]) for i in range(number_books)]
vec_seg_new	  = model2.transform(segments)
print "STEP 4 done"

'''#####################################'''
'''###############Step 5################'''
'''Applying SegmentElicitation Procedure'''
'''#####################################'''
'''
vec_seg_cls(dense)  = vector representation of each segment in corresponding cluster
vec_seg_new(dense) 	= vector representation of each segment
word_cls_frq = frequency of feature words(max_features=1500) in each cluster
			 = [[25,100,13,15,253,.... number of feature words] cluster 0
			    [65,200,123,10,15,.... number of feature words] cluster 1
			    ....
			    number of clusters/books
			   ]
word_frq 	 = each feature word(max_features=1500) frequency in whole document
			 = [150,550,260,1021,.... number of feature words(max_features=1500)]
post_p_w	 = posterior probability of each feature word in each cluster/book
			 = [ [0.3,0.25,.... number of clusters/books] word 1
				 [0.1,0.15,.... number of clusters/books] word 2
				 ....
				 number of feature words(max_features=1500)
			   ]
post_p_seg	 = posterior probability of each segment in each cluster
			 = [ [[0.85,0.01,0.1,... number of books,0(segment number)], [segment 2],.... number of segmensin this cluster] cluster 1 
			   	 [[0.85,0.01,0.1,... number of books,1(segment number)], [segment 2],.... number of segmensin this cluster] cluster 2 
			   	 ....
			   	 number of clusters/books
			   ]
best_seg	 = 80% of post_p_seg for each cluster in same format
			 = [ [[0.85,0.01,0.1,... number of books,0(segment number)], [segment 2],.... number of segmensin this cluster] cluster 1 
			   	 [[0.85,0.01,0.1,... number of books,1(segment number)], [segment 2],.... number of segmensin this cluster] cluster 2 
			   	 ....
			   	 number of clusters/books
			   ]
'''
	# calculating posterior probability of words
		# variables
post_p_w 	  = []
dense_array   = [i.toarray() for i in vec_seg_cls]
dense_array1  = vec_seg_new.toarray()
word_cls_frq  = [[sum(word_f) for word_f in zip(*cluster)] for cluster in dense_array]
word_frq 	  = [sum(word_f) for word_f in zip(*word_cls_frq)]
		# main
for i in range(max_features):
	post_p_w.append([])
	for j in range(number_books):
		post_p_w[i].append(float(word_cls_frq[j][i])/word_frq[i])
	# calculating posterior probability of segments in each cluster
post_p_seg = [[] for i in range(number_books)]
		# jth segment ith cluster
for j in range(number_seg):
	cls_num = label_p[j]
	temp = []
	for i in range(number_books):
		summation = 0
		for k in range(max_features):
			if (dense_array1[j][k]>0 and post_p_w[k][i]>0):
				summation += math.log(post_p_w[k][i])
		temp.append(summation)
	temp.append(j)
	post_p_seg[cls_num].append(temp)
	# print post_p_seg[cls_num][-1]
'''################finding vital segment for each cluster####################'''
'''Choosing best 80%(best_per) of segments to represent corresponding cluster'''
'''##########################################################################'''
best_seg = []
for i in range(number_books):
	end = int(best_per*len(post_p_seg[i]))
	sort_seg = sorted(post_p_seg[i], key=lambda x:-x[i]+max(x[:i]+x[i+1:-1]))
	best_seg.append(sort_seg[:end])
print "STEP 5 done"
'''######'''
'''Step 5'''
'''######'''

'''#################################################################################################'''
'''#########################################Step 6##################################################'''
'''Representing vital segments in form of minimum 3 frq feature words for each corresponding cluster'''
'''#################################################################################################'''
'''
vec_seg(dense) = vector representation of each segment
vital_seg = [ [ [0,1,1,0,0,1,1,1,0,0,0,... ~1500 max_features(=1500)], [0,1,1,0,0,1,1,1,0,0,0,...],.... number of vital segments] cluster 0
			  [ [0,1,1,0,0,1,1,1,0,0,0,... ~1500 max_features(=1500)], [0,1,1,0,0,1,1,1,0,0,0,...],.... number of vital segments] cluster 1
			  ....
			  number of clusters
			]
'''
print "STEP 5 done"

vital_seg = []
for cluster_n in range(number_books):
	vital_seg.append([])
	for seg in best_seg[cluster_n]:
		vital_seg[cluster_n].append(dense_array1[seg[-1]])
print "STEP 6 done"
'''######'''
'''Step 6'''
'''######'''

'''###############################################################################'''
'''#################################Step 7########################################'''
'''Training using Bernouli Naive-Bayesian model to learn a classifier on vital_seg'''
'''###############################################################################'''
'''
vital_seg = 2*number of vital_seg*1500
train = 2*number of vital_seg*1500
'''
train = []
labels= []
for cluster_n in range(number_books): 
	for seg in vital_seg[cluster_n]:
		train.append(seg.tolist())
		labels.append(cluster_n)
model3 = BNB(fit_prior = False)
model3 = model3.fit(train, labels)
print "STEP 7 done"
'''######'''
'''Step 7'''
'''######'''

'''################################################################'''
'''##########################Step 8################################'''
'''classfying sentences on trained classifier and calculating score'''
'''################################################################'''
vec_sen = model2.transform(merged_data[:1000])
temp = model3.predict_proba(vec_sen)
predicted = [map(lambda x: (x),temp[i]).index(max(temp[i])) for i in range(test_size)]
org_label = label_sen[:test_size]
print model3.score(vec_sen, org_label)
print "STEP 8 done"
'''######'''
'''Step 8'''
'''######'''