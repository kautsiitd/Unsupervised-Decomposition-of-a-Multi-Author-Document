import pickle

dataset_name = "Becker-Posner"
temp = dict()

with open("pos_tag.txt") as f:
	data = f.read().splitlines()

sentences = data[::2]
labels    = data[1::2]

for sentence,label in zip(sentences,labels):
	temp[sentence] = label

pickle.dump(temp,open("Pickle/"+dataset_name+"_pos.pickle",'wb'))