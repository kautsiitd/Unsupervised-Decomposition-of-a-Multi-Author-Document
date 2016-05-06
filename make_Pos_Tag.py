import pickle
import os
import io

dataset_name = "Becker-Posner"

with open("Pickle/"+dataset_name+"_pos.pickle") as f:
	temp = pickle.load(f)

folder 		= "dataset/Original/"+dataset_name
books_names = os.listdir(folder)

books_data 	= []
for book in books_names:
	path = os.path.join(folder,book)
	f    = io.open(path, encoding="ISO-8859-1")
	books_data.append(f.read().splitlines())

try:
	os.mkdir("dataset/Pos_Tag/"+dataset_name)
except:
	pass

for book_data,book_name in zip(books_data,books_names):
	f = open("dataset/Pos_Tag/"+dataset_name+"/"+book_name,'wb')
	for sentence in book_data:
		f.write(temp[str(sentence.encode("ISO-8859-1"))])
		f.write('\n')
	f.close()
