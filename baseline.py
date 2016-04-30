'''#############################'''
'''#####Importing Libraries#####'''
'''#############################'''
import sys
import os
import io
import re
from random import randint as rnd
from random import shuffle
from itertools import groupby
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.mixture import GMM
from pprint import pprint
import operator
import math
import nltk
from sklearn.naive_bayes import BernoulliNB as BNB
import itertools
print "Import Done"
'''#############################'''
'''#####Importing Libraries#####'''
'''#############################'''

# variables
V 		  = 200
	# number of most frequent features
max_limit = 1500
	# segment size
seg_size  = 30
	# choosing top vital segment in a class
best_per  = .8

'''###########################'''
'''##########Step 1###########'''
'''extracting and merging data'''
'''###########################'''
'''
books_names = ['a',b','c',....]
merged_data = ['sentence1','sentence2',.....]
label_sen   = [0,0,0,1,2,.....]								label of sentence in merged_data
segments    = ['segment1','segment2',.....]
label_in_seg= [[0,0,0,1,2,0,0,..],[0,1,1,2,1,0,1,..],....]	label of sentences in individual segments
label_seg   = [0,1,1,0,2,0,...]								book with max count in segment
'''
# file names
	# Eze-Job
	# NewYorkTimesArticles
	# MD-TF
	# GC-MD
	# Becker-Posner
folder 		= "dataset/GC-MD-TF"
books_names = os.listdir(folder)
merged_data	= []
label_sen	= []
segments 	= []
label_in_seg= []
label_seg 	= []
	# main
number_books= len(books_names)
books_data 	= []
for book in books_names:
	path = os.path.join(folder,book)
	f    = io.open(path, encoding="ISO-8859-1")
	books_data.append(f.readlines())
number_sen	= [len(book_data) for book_data in books_data]
total_sen	= sum(number_sen)
number_seg	= int(math.ceil(total_sen/seg_size))
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
model = model with feature words having atleast frequency = 3
vec_seg(sparse matrix) = [ [0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,... number of feature words]
						 [0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,... whether word present or not]
						 ....
						 number of segments
		  				 ]
number_f_w = number of feature words extracted from merged data
'''
model		  = CV(binary = True, min_df = 3)
model 		  = model.fit(merged_data)
vec_seg		  = model.transform(segments)
number_f_w	  = len(model.vocabulary_)
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
label_p = []
'''
model1		  = GMM(n_components = number_books, n_iter = 1000, covariance_type = 'diag', n_init = 5, verbose = 1)
model1		  = model1.fit(vec_seg.toarray())
label_p 	  = model1.predict_proba(vec_seg.toarray())
	# calculating recall
# recall_clusters= dict()
# temp = [[[0 ,0] ,[0 ,0]] for i in range(number_books)]
# for i in itertools.permutations(range(number_books)):
# 	temp_mapping = dict(zip(range(number_books),i))
# for i in range(len(label_p)):
# 	predicted = label_p[i]).index(max(label_p[i])
# 	given_l = org_label[i]
# 	if()
# 	if(label_p[i][0] == 1):
# 		if(float(label_seg[i].count(0))/len(label_seg[i]) > .99):
# 			recall_cluster [0][0] += 1
# 			recall_cluster1[1][1] += 1
# 		else:
# 			recall_cluster [0][1] += 1
# 			if(float(label_seg[i].count(1))/len(label_seg[i]) > .99):
# 				recall_cluster1[1][0] += 1
# 			else:
# 				recall_cluster1[1][1] += 1
# 	else:
# 		if(float(label_seg[i].count(1))/len(label_seg[i]) > .99):
# 			recall_cluster [1][0] += 1
# 			recall_cluster1[0][1] += 1
# 		else:
# 			recall_cluster [1][1] += 1
# 			if(float(label_seg[i].count(0))/len(label_seg[i]) > .99):
# 				recall_cluster1[0][0] += 1
# 			else:
# 				recall_cluster1[0][1] += 1
# print recall_cluster, recall_cluster1
# if sum(zip(*recall_cluster)[0]) < sum(zip(*recall_cluster1)[0]):
# 	mapping = [1,0]
# 	print "Recall: ",float(recall_cluster1[0][0])/org_seg[0],float(recall_cluster1[1][0])/org_seg[1]
# 	print "Pricision: ",float(recall_cluster1[0][0])/sum(recall_cluster1[0]),float(recall_cluster1[1][0])/sum(recall_cluster1[1])
# else:
# 	print "Recall: ",float(recall_cluster[0][0])/org_seg[0],float(recall_cluster[1][0])/org_seg[1]
# 	print "Pricision: ",float(recall_cluster[0][0])/sum(recall_cluster[0]),float(recall_cluster[1][0])/sum(recall_cluster[1])
	# putting segments in corresponding cluster
clusters	  = [[] for i in range(number_books)]
[clusters[map(lambda x: (x),label_p[i]) . index(max(label_p[i]))].append(segments[i]) for i in range(number_seg)]
print "STEP 3 done"
'''######'''
'''Step 3'''
'''######'''
sys.exit()

'''######'''
'''Step 4'''
'''######'''
# revectorising segments with max_limit most frequent words
model2		  = CV(max_features = max_limit)
model2 		  = model2.fit(merged_data)
vec_seg_cls   = [model2.transform(clusters[i]) for i in range(number_books)]
vec_seg_new	  = model2.transform(segments)
print "STEP 4 done"
'''######'''
'''Step 4'''
'''######'''

'''######'''
'''Step 5'''
'''######'''
# Applying SegmentElicitation Procedure
	# calculating posterior probability of words
		# variables
post_p_w 	  = []
dense_array   = [i.toarray() for i in vec_seg_cls]
dense_array1  = vec_seg_new.toarray()
word_cls_frq  = [[sum(word_f) for word_f in zip(*cluster)] for cluster in dense_array]
word_frq 	  = [sum(word_f) for word_f in zip(*word_cls_frq)]
		# main
for i in range(max_limit):
	post_p_w.append([])
	for j in range(number_books):
		post_p_w[i].append(float(word_cls_frq[j][i])/word_frq[i])
# for i in post_p_w:
# 	print i
	# calculating posterior probability of segments in each cluster
post_p_seg = [[] for i in range(number_books)]
		# jth segment ith cluster
for j in range(number_seg):
	cls_num = map(lambda x: (x),label_p[j]) . index(max(label_p[j]))
	temp = []
	for i in range(number_books):
		temp.append( sum([math.log(post_p_w[k][i]) for k in range(max_limit) if (dense_array1[j][k]>0 and post_p_w[k][i]>0) ]) )
	temp.append(j)
	post_p_seg[cls_num].append(temp)
	# print post_p_seg[cls_num][-1]
	# finding vital segment for each cluster
sort_seg = [sorted(post_p_seg[i], key=lambda x:-x[i]+max(x[:i]+x[i+1:-1]))[:int(best_per*len(post_p_seg[i]))] for i in range(number_books)]
print "STEP 5 done"
'''######'''
'''Step 5'''
'''######'''

'''######'''
'''Step 6'''
'''######'''
# representing vital segments in form of original minimum 3 frq words for each corresponding cluster
	# vital_seg = [cluster][seg][binary word representation]
dense_array2 = vec_seg.toarray()
vital_seg= [ [dense_array2[seg[-1]] for seg in sort_seg[cluster_n]] for cluster_n in range(number_books)]
print "STEP 6 done"
'''######'''
'''Step 6'''
'''######'''

'''######'''
'''Step 7'''
'''######'''
# Applying Bernouli Naive-Bayesian model to learn a classifier on vital_seg
train = []
labels= []
	# using nltk
		# for cluster_n in range(number_books): 
		# 	for seg in vital_seg[cluster_n]:
				# seg_l = seg.tolist()
				# temp = {i:seg_l[i] for i in range(number_f_w)}
			# train.append((temp ,cluster_n))
	# using sklearn
for cluster_n in range(number_books): 
	for seg in vital_seg[cluster_n]:
		train.append(seg.tolist())
		labels.append(mapping[cluster_n])
model3 = BNB(binarize = None, fit_prior = False)
model3 = model3.fit(train, labels)
print "STEP 7 done"
'''######'''
'''Step 7'''
'''######'''

'''######'''
'''Step 8'''
'''######'''
# classfying sentences on trained classifier and calculating score
	# using nltk
		# vec_sen = model.transform(merged_data).toarray()
		# for sen in vec_sen:
		# 	temp = {i:sen[i] for i in range(number_f_w)}
		# 	print classifier.prob_classify(temp).prob(0),classifier.prob_classify(temp).prob(1)
	# using sklearn
for label in label_sen:
	label = mapping[label]
		# for segments
test_size = len(vec_seg.toarray())
vec_sen = model.transform(merged_data)
# set_a   = zip(vec_sen,label_sen[:test_size])
# shuffle(set_a)
temp = model3.predict_proba(vec_seg)
predicted = [map(lambda x: (x),temp[i]).index(max(temp[i])) for i in range(test_size)]
# for i in range(test_size):
# 	print temp[i], org_label[i], map(lambda x: (x),temp[i]).index(max(temp[i]))
# print model3.score(zip(*set_a)[0], zip(*set_a)[1])
print model3.score(vec_seg, org_label)
print labels.count(0),labels.count(1),org_label.count(0),org_label.count(1),predicted.count(0),predicted.count(1)
print "STEP 8 done"
'''######'''
'''Step 8'''
'''######'''

# test_size = 1000
# vec_sen = model.transform(merged_data[:1000])
# temp = model3.predict_proba(vec_sen)
# predicted = [map(lambda x: (x),temp[i]).index(max(temp[i])) for i in range(test_size)]
# org_label = label_sen[:test_size]
# print model3.score(vec_sen, org_label)
# print labels.count(0),labels.count(1),org_label.count(0),org_label.count(1),predicted.count(0),predicted.count(1)
# print "STEP 8 done"
