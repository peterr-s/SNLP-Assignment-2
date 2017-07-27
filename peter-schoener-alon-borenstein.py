import transition
import conllu
import numpy
import sys
from sklearn import linear_model

def get_actions(sentences) :
	features = []
	transitions = []
	
	for sentence in sentences :
		s_f_list = []
		s_t_list = []
		#s = Sent(sentence)
		o = Oracle(sentence, False, True)
		c = Config(sentence)
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
			buffer_top = c.buffer[-3:]
			buffer_pos = ([None] * (3 - len(buffer_top))) + [pos[x] for x in buffer_top]
			buffer_form = ([None] * (3 - len(buffer_top))) + [form[x] for x in buffer_top]
			
			f_list = [None] * 12
			f_list[::2] = stack_pos + buffer_pos
			f_list[1::2] = stack_form + buffer_form
			
			s_f_list += [f_list]
			s_t_list += [(act, arg)]
			
			getattr(c, act)(arg)
		assert s == c.finish()
		features += [s_f_list]
		transitions += [s_t_list]
	
	return features, transitions

def make_model(sentences) :
	model = linear_model.SGDClassifier()
	train_model(model, sentences)
	return model

def train_model(model, sentences) :
	features, transitions = get_actions(sentences)
	x = numpy.array(features)
	y = numpy.array(transitions)
	model.fit(x, y)

if __name__ == "__main__" :
	if len(sys.argv) < 3 :
		print("syntax: python", argv[0], "train_set test_set")
		exit(1)

	tr_set = [sentence for sentence in load(argv[1])]

	model = make_model(tr_set)
	
	for sentence in load(argv[2]) :
		print(model.predict(sentence))
		print()
	
	exit(0)