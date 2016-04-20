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

# variables
V 		  = 200
	# number of most frequent features
max_limit = 1500
	# segment size
seg_size  = 30
	# choosing top vital segment in a class
best_per  = .8
	# mapping that GMM do or ith cluster represent mapping[i]th author
mapping   = [0,1]

# STEP 1
# extracting and merging data
	# variables
	# Eze-Job
	# NewYorkTimesArticles
	# MD-TF
	# GC-MD
folder 		= "dataset/Becker-Posner"
merged_data	= []
	# main
books_names = os.listdir(folder)
books_data 	= [io.open(os.path.join(folder,book), encoding="ISO-8859-1").readlines() for book in books_names]
number_books= len(books_names)
number_sen	= [len(book_data) for book_data in books_data]
total_sen	= sum(number_sen)
number_seg	= (total_sen/seg_size)+1
count_sen 	= [0]*number_books
label_sen	= []
label_seg	= []
while(sum(count_sen) != total_sen):
	book_num	  = rnd(1,number_books) - 1
	size		  = rnd(1,V)
	new_count_sen = count_sen[book_num] + min(size,number_sen[book_num]-count_sen[book_num])
	merged_data.extend( [re.sub('[\r\n]','',i) for i in books_data[book_num][count_sen[book_num]:new_count_sen]] )
	label_sen.extend( [book_num] * (new_count_sen - count_sen[book_num]) )
	count_sen[book_num]	= new_count_sen
segments = [' '.join(merged_data[ seg_size*i:min(seg_size*(i+1),total_sen) ]) for i in range(number_seg)]
label_seg= [label_sen[ seg_size*i:min(seg_size*(i+1),total_sen) ] for i in range((total_sen/seg_size)+1)]
print "STEP 1 done"
	# calculating segments by each author
org_seg		= [0 ,0 ,0]
for i in range(len(label_seg)):
	if float(label_seg[i].count(0))/len(label_seg[i]) == 1:org_seg[0] += 1
	elif float(label_seg[i].count(0))/len(label_seg[i]) == 0:org_seg[1] += 1
	else:org_seg[2] += 1
print "Author 1:",org_seg[0]
print "Author 2:",org_seg[1]
print "Mixed   :",org_seg[2]

# STEP 2
# finding features and vectorising segments
model		  = CV(binary = True, min_df = 3)
model 		  = model.fit(merged_data)
vec_seg		  = model.transform(segments)
number_f_w	  = len(model.vocabulary_)
print "STEP 2 done"

# STEP 3
# classifying into clusters using GMM
model1		  = GMM(n_components = number_books, n_iter = 1000, covariance_type = 'diag', n_init = 5, verbose = 1)
model1		  = model1.fit(vec_seg.toarray())
label_p 	  = model1.predict_proba(vec_seg.toarray())
	# calculating recall
recall_cluster = [[0 ,0] ,[0 ,0]]
recall_cluster1= [[0 ,0] ,[0 ,0]]
for i in range(len(label_p)):
	if(label_p[i][0] == 1):
		if(float(label_seg[i].count(0))/len(label_seg[i]) > .99):
			recall_cluster [0][0] += 1
			recall_cluster1[1][1] += 1
		else:
			recall_cluster [0][1] += 1
			if(float(label_seg[i].count(1))/len(label_seg[i]) > .99):
				recall_cluster1[1][0] += 1
			else:
				recall_cluster1[1][1] += 1
	else:
		if(float(label_seg[i].count(1))/len(label_seg[i]) > .99):
			recall_cluster [1][0] += 1
			recall_cluster1[0][1] += 1
		else:
			recall_cluster [1][1] += 1
			if(float(label_seg[i].count(0))/len(label_seg[i]) > .99):
				recall_cluster1[0][0] += 1
			else:
				recall_cluster1[0][1] += 1
print recall_cluster, recall_cluster1
if sum(zip(*recall_cluster)[0]) < sum(zip(*recall_cluster1)[0]):
	mapping = [1,0]
	print "Recall: ",float(recall_cluster1[0][0])/org_seg[0],float(recall_cluster1[1][0])/org_seg[1]
	print "Pricision: ",float(recall_cluster1[0][0])/sum(recall_cluster1[0]),float(recall_cluster1[1][0])/sum(recall_cluster1[1])
else:
	print "Recall: ",float(recall_cluster[0][0])/org_seg[0],float(recall_cluster[1][0])/org_seg[1]
	print "Pricision: ",float(recall_cluster[0][0])/sum(recall_cluster[0]),float(recall_cluster[1][0])/sum(recall_cluster[1])
	# putting segments in corresponding cluster
clusters	  = [[] for i in range(number_books)]
[clusters[map(lambda x: (x),label_p[i]) . index(max(label_p[i]))].append(segments[i]) for i in range(number_seg)]
print "STEP 3 done"

# STEP 4
# revectorising segments with max_limit most frequent words
model2		  = CV(max_features = max_limit)
model2 		  = model2.fit(merged_data)
vec_seg_cls   = [model2.transform(clusters[i]) for i in range(number_books)]
vec_seg_new	  = model2.transform(segments)
print "STEP 4 done"

# STEP 5
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

# STEP 6
# representing vital segments in form of original minimum 3 frq words for each corresponding cluster
	# vital_seg = [cluster][seg][binary word representation]
dense_array2 = vec_seg.toarray()
vital_seg= [ [dense_array2[seg[-1]] for seg in sort_seg[cluster_n]] for cluster_n in range(number_books)]
print "STEP 6 done"

# STEP 7
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

# STEP 8
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
org_label = [[0,1][float(label_seg[i].count(1))/len(label_seg[i])>.65] for i in range(test_size)]
# for i in range(test_size):
# 	print temp[i], org_label[i], map(lambda x: (x),temp[i]).index(max(temp[i]))
# print model3.score(zip(*set_a)[0], zip(*set_a)[1])
print model3.score(vec_seg, org_label)
print labels.count(0),labels.count(1),org_label.count(0),org_label.count(1),predicted.count(0),predicted.count(1)
print "STEP 8 done"

# test_size = 1000
# vec_sen = model.transform(merged_data[:1000])
# temp = model3.predict_proba(vec_sen)
# predicted = [map(lambda x: (x),temp[i]).index(max(temp[i])) for i in range(test_size)]
# org_label = label_sen[:test_size]
# print model3.score(vec_sen, org_label)
# print labels.count(0),labels.count(1),org_label.count(0),org_label.count(1),predicted.count(0),predicted.count(1)
# print "STEP 8 done"
