'''############################'''
'''#####Defining Variables#####'''
##################################
V 		  = 200
b_num     = 0
b = ["Becker-Posner","GC-TF-PK","MD-TF-PK","MD-GC-PK","MD-GC-TF-PK"]
gmm_initialisation = 5
	# segment size
seg_size  = 30
	# number of sentence to test on from final model
n_gram_size = 3
'''############################'''
'''#####Defining Variables#####'''
'''############################'''

'''##########Step 1###########'''
'''extracting and merging data'''
#################################
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
'''###########################'''

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
//Code

###################STEP2##################
'''Find If given Segment is Pure or Mix'''
##########################################
'''
Calculate Similiarity Index of Segment:
	Higher Similiarity Index, Higher Pure
'''
mixed_segments = []
pure_segments  = []
sentence_size  = []
threshold	   = 1
# calculating sentence sizes in each segment
for segment in segments_parser:
	sentence_size.append([])
	for sentence in segment:
		sentence_size[-1].append(len(sentence))

# calculating similiarity index for each segment
number_seg = len(segments_parser)
for i in range(number_seg):
	segment = segments_parser[i]
	similiarity_index = 0
	seg_size = len(segment)
	'''iterating over sentences in a segment'''
	for j in range(seg_size):
		for pq_gram in segment[j]:
			'''checking current pqgram is in how 
			many other sentences of same segment'''
			for k in range(j,seg_size):
				sentence_size = len(segment[k])
				if pq_gram in segment[k]:
					similiarity_index += 1.0/(sentence_size[i][k]*sentence_size[i][k])
	print similiarity_index
	if similiarity_index > threshold:
		pure_segments.append(segments_parser[i])
	else:
		mixed_segments.append(segments_parser[i])