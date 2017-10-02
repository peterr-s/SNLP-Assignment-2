#!/usr/bin/python3

# Authors:	Peter Schoener, 4013996
#			Alon Borenstein, 4041104
# Honor Code: We pledge that this program represents our own work.

import numpy
import os, sys
from sklearn import svm, metrics
import gensim

## CONLLU
# for loading and saving conllu files.

cols = 'id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'

def get_dimensionality(model) :
 	for k, v in model.vocab.items() : # this should only reach the first item before returning but it seemed like the most elegant way to get the "first" item of the keyset
 		return len(model[k])

class Sent(object):
	"""a conllu sentence.

	dumb: dummy value for the pseudo root node

	cols: names of the 10 columns in the conllu format

	id: tuple<int>

	head: tuple<int|str>

	form, lemma, upostag, xpostag, feats, deprel, deps, misc: tuple<str>

	all 10 tuples are of equal length.

	multi: multi-word tokens.

	"""
	
	__slots__ = cols + ('multi',)
	cols = cols
	dumb = ""
		
	def __init__(self, lines):
		"""lines: iter<str>"""
		dumb = self.dumb
		multi = []
		nodes = [[0, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb]]
		for line in lines:
			node = line.split("\t")
			assert 10 == len(node)
			try:
				node[0] = int(node[0])
			except ValueError:
				if "-" in node[0]: multi.append(line)
			else:
				try:  # head might be empty for interim results
					node[6] = int(node[6])
				except ValueError:
					pass
				nodes.append(node)
		self.multi = tuple(multi)
		for attr, val in zip(self.cols, zip(*nodes)):
			setattr(self, attr, val)

	def __eq__(self, other):
		for attr in self.__slots__:
			if ((not hasattr(other, attr))
				or (getattr(self, attr) != getattr(other, attr))):
				return False
		else:
			return True

del cols


def load(file_path):
	"""-> iter<Sent>"""
	with open(file_path, encoding='utf-8') as file:
		sent = []
		for line in file:
			line = line.strip()
			if line.startswith("#"):
				pass
			elif line:
				sent.append(line.replace(" ", "\xa0"))
			elif sent:
				yield Sent(sent)
				sent = []
		if sent: yield Sent(sent)


def save(sents, file_path):
	"""sents: iter<Sent>"""
	with open(file_path, 'w', encoding='utf-8') as file:
		for sent in sents:
			multi_idx = [int(multi[:multi.index("-")]) for multi in sent.multi]
			w, m = 1, 0
			while w < len(sent.id):
				if m < len(multi_idx) and w == multi_idx[m]:
					line = sent.multi[m]
					m += 1
				else:
					line = "\t".join([str(getattr(sent, col)[w]) for col in sent.cols])
					w += 1
				file.write(line.replace("\xa0", " "))
				file.write("\n")
			file.write("\n")

## TRANSITION

# an implementation of transition-based projective/non-projective
# dependency-parsing algorithms, with a static oracle.

from itertools import repeat
from bisect import bisect_left, insort_right
from copy import copy

class Config(object):
	"""a transition configuration.

	sent: Sent

	input: list representing reversed buffer β

	stack: list representing reversed stack σ

	graph: adjacency list representation of the current graph

	deprel: arc labels of the current graph

	------------------------- example -------------------------

	config c: Config

	action a: 'shift' | 'right' | 'left' | 'swap'

	deprel l: str | None

	current active nodes: `i, j = c.stack_nth(2), c.stack_nth(1)`

	advance in transition: `if c.doable(a): getattr(c, a)(l)`

	finish transition: `if c.is_terminal(): parsed_sent = c.finish()`

	"""
	__slots__ = 'sent', 'input', 'stack', 'graph', 'deprel'

	def __init__(self, sent):
		""""sent: Sent"""
		n = len(sent.head)
		self.sent = sent
		self.input = list(range(n - 1, 0, -1))
		self.stack = [0]
		self.graph = [[] for _ in range(n)]
		self.deprel = list(repeat(Sent.dumb, n))

	def is_terminal(self):
		"""-> bool"""
		return not self.input and 1 == len(self.stack)

	def stack_nth(self, n):
		"""returns the id of the nth (n >= 1) stack node. """
		return self.stack[-n]

	def input_nth(self, n):
		"""returns the id of the nth (n >= 1) input node."""
		return self.input[-n]

	def doable(self, act):
		"""-> bool; act: 'shift' | 'right' | 'left' | 'swap'"""
		if 'shift' == act:
			return 0 != len(self.input)
		elif 'right' == act:
			return 2 <= len(self.stack)
		elif 'left' == act:
			return 2 <= len(self.stack) and 0 != self.stack[-2]
		elif 'swap' == act:
			return 2 <= len(self.stack) and 0 < self.stack[-2] < self.stack[-1]
		else:
			raise TypeError("unknown act: {}".format(act))

	def shift(self, _=None):
		"""(σ, i|β, A) ⇒ (σ|i, β, A)"""
		self.stack.append(self.input.pop())

	def right(self, deprel):
		"""(σ|i|j, β, A) ⇒ (σ|i, B, A ∪ {(i, l, j)})"""
		j = self.stack.pop()
		i = self.stack[-1]
		insort_right(self.graph[i], j)
		# i -deprel-> j
		self.deprel[j] = deprel

	def left(self, deprel):
		"""(σ|i|j, β, A) ⇒ (σ|j, β, A ∪ {(j, l, i)})"""
		j = self.stack[-1]
		i = self.stack.pop(-2)
		insort_right(self.graph[j], i)
		# i <-deprel- j
		self.deprel[i] = deprel

	def swap(self, _=None):
		"""(σ|i|j, β, A) ⇒ (σ|j, i|β, A)"""
		self.input.append(self.stack.pop(-2))

	def finish(self):
		"""-> Sent"""
		graph = self.graph
		deprel = self.deprel
		if 1 < len(graph[0]):  # ensure single root
			h = graph[0][0]
			for d in graph[0][1:]:
				insort_right(graph[h], d)
				deprel[d] = 'parataxis'
		head = list(repeat(Sent.dumb, len(graph)))
		for h, ds in enumerate(graph):
			for d in ds:
				head[d] = h
		sent = copy(self.sent)
		sent.head = tuple(head)
		sent.deprel = tuple(deprel)
		return sent


class Oracle(object):
	"""three possible modes:

	0. proj=True, arc-standard

	1. lazy=False, non-proj with swap (Nivre 2009)

	2. default, lazy swap (Nivre, Kuhlmann, Hall 2009)

	------------------------- example -------------------------

	see `test_oracle` in `__main__`.

	"""
	__slots__ = 'sent', 'mode', 'graph', 'order', 'mpcrt'

	def __init__(self, sent, proj=False, lazy=True):
		"""sent: Sent, proj: bool, lazy: bool"""
		n = len(sent.head)
		self.sent = sent
		self.mode = 0
		self.graph = [[] for _ in range(n)]
		for i in range(1, n):
			self.graph[sent.head[i]].append(i)
		if proj: return
		self.mode = 1
		self.order = list(range(n))
		self._order(0, 0)
		if not lazy: return
		self.mode = 0
		self.mpcrt = list(repeat(None, n))
		config = Config(sent)
		while not config.is_terminal():
			act, arg = self.predict(config)
			if not config.doable(act):
				break
			getattr(config, act)(arg)
		self._mpcrt(config.graph, 0, 0)
		self.mode = 2

	def _order(self, n, o):
		# in-order traversal ordering
		i = bisect_left(self.graph[n], n)
		for c in self.graph[n][:i]:
			o = self._order(c, o)
		self.order[n] = o
		o += 1
		for c in self.graph[n][i:]:
			o = self._order(c, o)
		return o

	def _mpcrt(self, g, n, r):
		# maximal projective component root
		self.mpcrt[n] = r
		i = 0
		for c in self.graph[n]:
			if self.mpcrt[c] is None:
				i = bisect_left(g[n], c, i)
				self._mpcrt(g, c, r if i < len(g[n]) and c == g[n][i] else c)

	def predict(self, config):
		"""Config -> (action, deprel): ('shift' | 'swap', None) | ('right' | 'left', str)"""
		if 1 == len(config.stack):
			return 'shift', None
		i, j = config.stack[-2:]
		if 0 != self.mode and self.order[i] > self.order[j]:
			if 1 == self.mode:
				return 'swap', None
			if not config.input or self.mpcrt[j] != self.mpcrt[config.input[-1]]:
				return 'swap', None
		if self.sent.head[i] == j and len(self.graph[i]) == len(config.graph[i]):
			return 'left', self.sent.deprel[i]
		if i == self.sent.head[j] and len(self.graph[j]) == len(config.graph[j]):
			return 'right', self.sent.deprel[j]
		return 'shift', None

## MAIN

def get_actions(sentences, embedding_model) :
	features = []
	transitions = []
	
	dim = get_dimensionality(embedding_model)
	
	for s in sentences :
		s_f_list = []
		s_t_list = []
		o = Oracle(s, False, True)
		c = Config(s)
		while not c.is_terminal():
			act, arg = o.predict(c)
			assert c.doable(act)
			
			# get lists of all forms and tags
			# note that everything is 1-indexed because of the root node (index 0 is empty in both lists)
			pos = getattr(s, "upostag")
			form = getattr(s, "form")
			
			# get stack top forms and tags, null-pad beginning for const length (3)
			stack_top = c.stack[-3:]
			stack_pos = ([None] * (3 - len(stack_top))) + [pos[x] for x in stack_top]
			stack_form = ([None] * (3 - len(stack_top))) + [form[x] for x in stack_top]
			
			# get buffer top forms and tags, null-pad beginning for const length (3)
			buffer_top = c.input[-3:][::-1] # reverse this because it's populated backwards
			buffer_pos = ([None] * (3 - len(buffer_top))) + [pos[x] for x in buffer_top]
			buffer_form = ([None] * (3 - len(buffer_top))) + [form[x] for x in buffer_top]
			
			f_list = [None] * 12
			f_list[::2] = stack_pos + buffer_pos
			f_list[1::2] = stack_form + buffer_form
			
			s_f_list += [numpy.concatenate([embedding_model[w] if w in embedding_model else numpy.zeros(dim) for w in f_list])]
			#s_t_list += [(act, arg)]
			s_t_list += [act]
			
			getattr(c, act)(arg)
		assert s == c.finish()
		features += s_f_list
		transitions += s_t_list

	return features, transitions

def make_model(sentences, embedding_model) :
	model = svm.LinearSVC()
	train_model(model, sentences, embedding_model)
	return model

def train_model(model, sentences, embedding_model) :
	features, transitions = get_actions(sentences, embedding_model)
	x = numpy.array(features)
	y = numpy.array(transitions)
	model.fit(x, y)

def predict_parse(s, model, embedding_model) :
	# get predicted sequence
	p_f_list = []
	p_t_list = []
	o = Oracle(s, False, True)
	c = Config(s)
	while not c.is_terminal() :
		# get lists of all forms and tags
		# note that everything is 1-indexed because of the root node (index 0 is empty in both lists)
		pos = getattr(s, "upostag")
		form = getattr(s, "form")
		
		# get stack top forms and tags, null-pad beginning for const length (3)
		stack_top = c.stack[-3:]
		stack_pos = ([None] * (3 - len(stack_top))) + [pos[x] for x in stack_top]
		stack_form = ([None] * (3 - len(stack_top))) + [form[x] for x in stack_top]
		
		# get buffer top forms and tags, null-pad beginning for const length (3)
		buffer_top = c.input[-3:][::-1] # reverse this because it's populated backwards
		buffer_pos = ([None] * (3 - len(buffer_top))) + [pos[x] for x in buffer_top]
		buffer_form = ([None] * (3 - len(buffer_top))) + [form[x] for x in buffer_top]
		
		f_list = [None] * 12
		f_list[::2] = stack_pos + buffer_pos
		f_list[1::2] = stack_form + buffer_form
		
		p_act = model.predict([numpy.concatenate([embedding_model[w] if w in embedding_model else numpy.zeros(dim) for w in f_list])])[0]
		a = 0
		while not c.doable(p_act) :
			if a >= 4 :
				print("no possible actions")
				break
			p_act = actions[a]
			a += 1
		
		if p_act == "right" :
			i = c.stack[-1]
			p_arg = o.sent.deprel[i]
		elif p_act == "left" :
			i = c.stack[-2]
			p_arg = o.sent.deprel[i]
		else :
			p_arg = None
		
		p_t_list += [p_act]
		getattr(c, p_act)(p_arg)
	
	s = c.finish()
	return s, p_t_list

if __name__ == "__main__" :
	if len(sys.argv) < 3 :
		print("syntax: python", sys.argv[0], "train_set test_set [embeddings]")
		exit(1)
	
	elif len(sys.argv) < 4 :
		sys.argv += ["embeddings.bin"]

	try :
		embedding_model = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[3]).wv
	except Exception : # UnicodeDecodeError, but there might be others
		embedding_model = gensim.models.Word2Vec.load(sys.argv[3]).wv
	
	tr_set = load(sys.argv[1])
	
	model = make_model(tr_set, embedding_model)
	
	test_set = load(sys.argv[2])
	actions = ["shift", "left", "right", "swap"]
	dim = get_dimensionality(embedding_model)
	acc_list = []
	for s in test_set :
		# get gold standard sequence
		g_t_list = []
		o = Oracle(s, False, True)
		c = Config(s)
		while not c.is_terminal():
			act, arg = o.predict(c)
			assert c.doable(act)
			
			# get lists of all forms and tags
			# note that everything is 1-indexed because of the root node (index 0 is empty in both lists)
			pos = getattr(s, "upostag")
			form = getattr(s, "form")
			
			# get stack top forms and tags, null-pad beginning for const length (3)
			stack_top = c.stack[-3:]
			stack_pos = ([None] * (3 - len(stack_top))) + [pos[x] for x in stack_top]
			stack_form = ([None] * (3 - len(stack_top))) + [form[x] for x in stack_top]
			
			# get buffer top forms and tags, null-pad beginning for const length (3)
			buffer_top = c.input[-3:][::-1] # reverse this because it's populated backwards
			buffer_pos = ([None] * (3 - len(buffer_top))) + [pos[x] for x in buffer_top]
			buffer_form = ([None] * (3 - len(buffer_top))) + [form[x] for x in buffer_top]
			
			f_list = [None] * 12
			f_list[::2] = stack_pos + buffer_pos
			f_list[1::2] = stack_form + buffer_form
			
			g_t_list += [act]
			
			getattr(c, act)(arg)
		assert s == c.finish()
		
		p_s, p_t_list = predict_parse(s, model, embedding_model)
		
		# compare gold standard and prediction
		l_attachment = 0.0
		u_attachment = 0.0
		sent_len = len(s.head)
		for n in range(sent_len) :
			if s.head[n] == p_s.head[n] :
				u_attachment += 1.0
				if s.deprel[n] == p_s.deprel[n] :
					l_attachment += 1.0
		l_attachment /= sent_len
		u_attachment /= sent_len
		acc_list += [[l_attachment, u_attachment]]
	print(numpy.mean(acc_list, axis = 0))
	
	exit(0)

# peak unlabelled attachment: 52.689%
#        labelled attachment: 49.304%