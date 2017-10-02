#!/usr/bin/python3

from transition import *
from conllu import *
import conllu
import numpy
import os, sys
from sklearn import linear_model, metrics
import gensim

def get_dimensionality(model) :
	for k, v in model.vocab.items() : # this should only reach the first item before returning but it seemed like the most elegant way to get the "first" item of the keyset
		return len(model[k])

def get_actions(sentences, embedding_model) :
	features = []
	transitions = []
	
	dim = get_dimensionality(embedding_model)
	
	for s in sentences :
		s_f_list = []
		s_t_list = []
		#s = Sent(sentence)
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

	# print(features[:10])
	# print(transitions[:10])
	return features, transitions

def make_model(sentences, embedding_model) :
	model = linear_model.SGDClassifier()
	train_model(model, sentences, embedding_model)
	return model

def train_model(model, sentences, embedding_model) :
	features, transitions = get_actions(sentences, embedding_model)
	x = numpy.array(features)
	y = numpy.array(transitions)
	model.fit(x, y)

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
	
	tr_set = conllu.load(sys.argv[1])
	
	model = make_model(tr_set, embedding_model)
	
	test_set = conllu.load(sys.argv[2])
	actions = ["shift", "left", "right", "swap"]
	dim = get_dimensionality(embedding_model)
	acc_list = []
	for s in test_set :
		# get gold standard sequence
		g_f_list = []
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
			
		#	g_f_list += [numpy.concatenate([embedding_model[w] if w in embedding_model else numpy.zeros(dim) for w in f_list])]
			g_t_list += [act]
			
			getattr(c, act)(arg)
		assert s == c.finish()
		
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
			
			#print(p_act)
			p_t_list += [p_act]
			getattr(c, p_act)(p_arg)
		
		# compare gold standard and prediction
		len_diff = len(p_t_list) - len(g_t_list)
		if len_diff > 0 :
			g_t_list += len_diff * ["NONE"]
		elif len_diff < 0 :
			p_t_list += (-len_diff) * ["NONE"]
		accuracy = metrics.accuracy_score(g_t_list, p_t_list)
	#	print(accuracy)
		acc_list += [accuracy]
	#print()
	print(numpy.mean(accuracy))
	
	exit(0)