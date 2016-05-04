from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger('/home/dell/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger')


'''Tag all the sentences in list and return list of tagged sentences'''
'''
data   = ['Sentence 1','Sentence 2',...... number of sentences]
tagged = ['tag1 tag2 ...', 'tag1 tag2 ...', .... number of sentences]
'''

def pos_tagging(data):
	i = 0
	print len(data)
	tagged = []
	for sentence in data:
		print i
		temp = []
		for key,value in st.tag(sentence.split()):
			temp.append(value)
		tagged.append(' '.join(temp))
		i += 1
	return tagged
