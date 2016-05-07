# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import sys
import io
import copy
import nltk
from nltk.internals import find_jars_within_path
from nltk.parse.stanford import StanfordParser
parser=StanfordParser(model_path="stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
stanford_dir = parser._classpath[0].rpartition('/')[0]
parser._classpath = tuple(find_jars_within_path(stanford_dir))
# from set_parser import parse_it
class Node(object):
	"""
		A generic representation of a tree node. Includes a string label and a list of a children.
	"""

	def __init__(self, label):
		"""
			Creates a node with the given label. The label must be a string for use with the PQ-Gram
			algorithm.
		"""
		self.label = label
		self.children = list()

	def addkid(self, node, before=False):
		"""
			Adds a child node. When the before flag is true, the child node will be inserted at the
			beginning of the list of children, otherwise the child node is appended.
		"""
		if before:  self.children.insert(0, node)
		else:   self.children.append(node)
		return self

class ShiftRegister(object):
	"""
		Represents a register which acts as a fixed size queue. There are only two valid
		operations on a ShiftRegister: shift and concatenate. Shifting results in a new
		value being pushed onto the end of the list and the value at the beginning list being
		removed. Note that you cannot recover this value, nor do you need to for generating
		PQ-Gram Profiles.
	"""

	def __init__(self, size):
		"""
			Creates an internal list of the specified size and fills it with the default value
			of "*". Once a ShiftRegister is created you cannot change the size without 
			concatenating another ShiftRegister.
		"""
		self.register = list()
		for i in range(size):
			self.register.append("*")
		
	def concatenate(self, reg):
		"""
			Concatenates two ShiftRegisters and returns the resulting ShiftRegister.
		"""
		temp = list(self.register)
		temp.extend(reg.register)
		return temp
	
	def shift(self, el):
		"""
			Shift is the primary operation on a ShiftRegister. The new item given is pushed onto
			the end of the ShiftRegister, the first value is removed, and all items in between shift 
			to accomodate the new value.
		"""
		self.register.pop(0)
		self.register.append(el)

class Profile(object):
	"""
		Represents a PQ-Gram Profile, which is a list of PQ-Grams. Each PQ-Gram is represented by a
		ShiftRegister. This class relies on both the ShiftRegister and tree.Node classes.
	"""
	
	def __init__(self, root, p=2, q=3):
		"""
			Builds the PQ-Gram Profile of the given tree, using the p and q parameters specified.
			The p and q parameters do not need to be specified, however, different values will have
			an effect on the distribution of the calculated edit distance. In general, smaller values
			of p and q are better, though a value of (1, 1) is not recommended, and anything lower is
			invalid.
		"""
		super(Profile, self).__init__()
		ancestors = ShiftRegister(p)
		self.list = list()
		
		self.profile(root, p, q, ancestors)
		self.sort()
	
	def profile(self, root, p, q, ancestors):
		"""
			Recursively builds the PQ-Gram profile of the given subtree. This method should not be called
			directly and is called from __init__.
		"""
		ancestors.shift(root.label)
		siblings = ShiftRegister(q)
		
		if(len(root.children) == 0):
			self.append(ancestors.concatenate(siblings))
		else:
			for child in root.children:
				siblings.shift(child.label)
				self.append(ancestors.concatenate(siblings))
				self.profile(child, p, q, copy.deepcopy(ancestors))
			for i in range(q-1):
				siblings.shift("*")
				self.append(ancestors.concatenate(siblings))

	def sort(self):
		"""
			Sorts the PQ-Grams by the concatenation of their labels. This step is automatically performed
			when a PQ-Gram Profile is created to ensure the intersection algorithm functions properly and
			efficiently.
		"""
		self.list.sort(key=lambda x: ''.join)

	def append(self, value):
		self.list.append(value)



def make_tree(tree):
	label = tree.split(',',1)[0][5:]
	temp = Node(label)
	list_children = tree.split(',',1)[1][2:-2]
	children = []
	stack = 0
	temp_str = ""
	for i in list_children:
		temp_str += i
		if(i == '('): stack+=1
		elif(i == ')'): stack-=1
		if(stack == 0):
			if(i == ')'):
				children.append("Tree"+temp_str)
			temp_str = ""
			continue
	for child in children:
		temp.addkid(make_tree(child))
	return temp

def find_pq_grams(sentence):
	temp1 = parser.raw_parse(sentence.decode("ISO-8859-1"))
	tree = str(list(temp1))[1:-1]
	temp = Profile(make_tree(tree)).list
	for i in range(len(temp)):
		for j in range(5):
			temp[i][j] = temp[i][j].strip("'")
	return temp

b_num     = 0
b = ["Becker-Posner","GC-TF-PK","MD-TF-PK","MD-GC-PK","MD-GC-TF-PK"]

folder 		= "dataset/Original/"+b[b_num]
books_names = os.listdir(folder)
books_data 	= []

for book in books_names:
	path = os.path.join(folder,book)
	f    = io.open(path, encoding="ISO-8859-1")
	books_data.append(f.readlines())

for book_data,book_name in zip(books_data,books_names):
	folder = "Backup/Parser/"+b[b_num]
	try:
		os.mkdir(folder)
	except:
		pass
	f = open(folder+"/"+book_name, 'wb')
	g = open(folder+"/"+book_name[:-4]+".error", 'wb')
	i = 0
	for sentence in book_data:
		sentence = str(sentence.encode("ISO-8859-1").strip())
		print i
		# temp = find_pq_grams(sentence)
		# temp = [item.encode("utf-8") for sublist in temp for item in sublist]
		# temp = ' '.join(temp)
		# f.write(sentence)
		# f.write('\n')
		# f.write(temp)
		# f.write('\n')
		try:
			temp = find_pq_grams(sentence)
			temp = [item.encode("utf-8") for sublist in temp for item in sublist]
			temp = ' '.join(temp)
			f.write(sentence)
			f.write('\n')
			f.write(temp)
			f.write('\n')
		except:
			print "kanu",sentence
			g.write("1")
			g.write(sentence)
			g.write('\n')
		i += 1
