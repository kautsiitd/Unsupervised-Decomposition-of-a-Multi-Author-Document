from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger('/home/dell/Desktop/Unsupervised-Decomposition-of-a-Multi-Author-Document/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger')
print st.tag('What is the airspeed of an unladen swallow ?'.split())