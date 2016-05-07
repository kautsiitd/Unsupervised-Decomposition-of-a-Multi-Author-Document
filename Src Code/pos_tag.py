import os
import sys
import subprocess
import shlex
from nltk.tag import StanfordPOSTagger
# cmd1 = "export STANFORDTOOLSDIR=/home/dell/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document"
# args = shlex.split(cmd1)
# subprocess.Popen(args, stdout=subprocess.PIPE,
#                         env={'STANFORDTOOLSDIR': '/home/dell/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document'})
# cmd1 = "export CLASSPATH=$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/stanford-postagger.jar:$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/models"
# args = shlex.split(cmd1)
# subprocess.Popen(args, stdout=subprocess.PIPE,
#                         env={'CLASSPATH': '$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/stanford-postagger.jar:$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/models'})
# os.system("export STANFORDTOOLSDIR=$HOME/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document")
# os.system("export CLASSPATH=$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/stanford-postagger.jar:$STANFORDTOOLSDIR/stanford-postagger-2015-12-09/models")

st = StanfordPOSTagger('/home/dell/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger')


'''Tag all the sentences in list and return list of tagged sentences'''
'''
data   = ['Sentence 1','Sentence 2',...... number of sentences]
tagged = ['tag1 tag2 ...', 'tag1 tag2 ...', .... number of sentences]
'''

def pos_tagging(data):
	f = open('../pos_tag.txt', 'wb')
	g = open('../errors.txt', 'wb')
	i = 0
	print len(data)
	tagged = []
	for sentence in data:
		print i
		temp = []
		try:
			temp1= st.tag(sentence.encode("ISO-8859-1").split())
			for key,value in temp1:
				temp.append(value)
			tagged.append(' '.join(temp))
			f.write(sentence.encode("ISO-8859-1"))
			f.write('\n')
			f.write(' '.join(temp))
			f.write('\n')
		except:
			print "kanu"
			g.write(sentence.encode("ISO-8859-1"))
			g.write('\n')
		i += 1
	return tagged
