#!/usr/bin/env python3

import sys
from collections import namedtuple

Token = namedtuple(
	'Token', "tid, form lemma pos xpos feats head deprel deps misc children")

def read_conllu(fname=None, fp=sys.stdin, mark_children=False):
	if fname is not None:
		fp = open(fname, 'r')

	treebank = []
	sent_start = True
	for line in fp:
		if line.startswith('#'):
			continue
		line = line.strip()

		if len(line) == 0 and not sent_start:
			if mark_children:
				for tok in sent:
					if tok.head is not None:
						hd = sent[tok.head]
						hd.children.append(tok.tid)
			treebank.append(sent)
			sent_start = True
			continue

		if mark_children: chi = []
		else: chi = None

		if sent_start:
			sent = [Token(
				0, "_", "root", "_", "_", "_", None, "_", "_", "_", chi)]
			sent_start = False

		(tid, form, lemma, pos, xpos, feats, head, deprel, deps, misc) = \
				line.strip().split('\t')
		if "-" in tid:
			continue
		sent.append(Token(int(tid),
						  form,
						  lemma,
						  pos,
						  xpos,
						  feats,
						  int(head),
						  deprel.split(":")[0],
						  deps,
						  misc,
						  chi))
	return treebank

# Main

if len(sys.argv) != 3:
	print("Usage: {} parser_output gold_standard".format(sys.argv[0]))
	sys.exit(-1)

out = read_conllu(sys.argv[1])
gs  = read_conllu(sys.argv[2])

if len(out) != len(gs):
	print("The number of sentences differ!")
	sys.exit(-1)

# arcs_lmatch_s = 0
# arcs_umatch_s = 0

arcs_lmatch_w = 0
arcs_umatch_w = 0
arcs_total = 0
for i in range(len(out)):
	sent_out = out[i]
	sent_gs = gs[i]

	if len(sent_out) != len(sent_gs):
		print("The number of words differ in sentence {}".format(i))
		sys.exit(-1)

	arcs_lmatch_sent = 0
	arcs_umatch_sent = 0
	ntokens = len(sent_out) - 1
	for j in range(1,len(sent_out)):
		if sent_out[j].head == sent_gs[j].head:
			arcs_umatch_sent += 1
			if sent_out[j].deprel == sent_gs[j].deprel:
				arcs_lmatch_sent += 1
	arcs_total += ntokens
	arcs_lmatch_w += arcs_lmatch_sent
	arcs_umatch_w += arcs_umatch_sent
	# arcs_lmatch_s += arcs_lmatch_sent / ntokens
	# arcs_umatch_s += arcs_umatch_sent / ntokens


# print("wUAS/wLAS/sUAS/sLAS: {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
#		 100 * arcs_umatch_w / arcs_total,
#		 100 * arcs_lmatch_w / arcs_total,
#		 100 * arcs_umatch_s / len(out),
#		 100 * arcs_lmatch_s / len(out))
# )

print("UAS: {:.2f}\tLAS: {:.2f}".format(
	100 * arcs_umatch_w / arcs_total,
	100 * arcs_lmatch_w / arcs_total)
)
