import os
import sys
import glob

dataset = "GC-TF-PK"
source_folder  = "../Backup/Parser/"+dataset
destination_folder = "../dataset/Parser/"+dataset

try:
	os.mkdir(destination_folder)
except:
	pass

files   = [f for f in os.listdir(source_folder) if f.endswith(".txt")]
for file_name in files:
	source = os.path.join(source_folder,file_name)
	with open(source) as f:
		data = f.readlines()[1::2]

	destination = os.path.join(destination_folder,file_name)
	with open(destination,"wb") as f:
		for all_pq in data:
			all_pq = all_pq.split()
			size = len(all_pq)
			pqs  = [all_pq[i:i+5] for i in range(0,size,5)]
			for pq in pqs:
				f.write(''.join(pq))
				f.write(" ")
			f.write('\n')