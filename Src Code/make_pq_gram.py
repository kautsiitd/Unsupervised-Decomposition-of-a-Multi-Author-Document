import os
import sys

dataset = "Becker-Posner"
source_folder  = "../Backup/Parser/"+dataset
destination_folder = "../dataset/Parser/"+dataset

files   = os.listdir(folder)
for file_name in files:
	source = os.join(source_folder,file_name)
	with open(source) as f:
		data = f.readlines()

	destination = os.join(destination_folder,file_name)
	with open(destination) as f:
		for all_pq in data:
			size = len(all_pq)
			pqs  = [all_pq[i:i+5].strip() for i in range(0,size,5)]
			pqs  = map(str,pqs)
			print pqs
			sys.exit()
