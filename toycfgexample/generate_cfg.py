from nltk.parse.generate import generate
from nltk import CFG, ngrams
import numpy as np
import itertools

CFG_STRING = '''
S -> S M T | T
T -> B1 S B2 | O
B1 -> '('
B2 -> ')'
O -> '1'
M -> '*'
'''
grammar = CFG.fromstring(CFG_STRING)
tokens = ['(', ')', '1', '*']
SEPARATOR = "-" * 40


def generate_all_possible_sentences():
	all_sentences = []
	for sentence in generate(grammar, depth=9):
		all_sentences.append(' '.join(sentence))

	return all_sentences


def freq_to_probabilities(n, givens, freq):
	P = np.zeros((len(givens), len(tokens)))

	# populate matrix with counts
	for row in range(len(givens)):
		for col in range(len(tokens)):
			obs = tuple(list(givens[row]) + list(tokens[col]))
			
			if obs in freq:
				P[row, col] = freq[obs]

		# normalize row
		if np.sum(P[row]) > 0:
			P[row] /= np.sum(P[row])

	return P


def print_transition_matrix(givens, P):
	print('{:6}\t{:6}\t{:6}\t{:6}\t{:6}'.format('', tokens[0].center(6), tokens[1].center(6), tokens[2].center(6), tokens[3].center(6)))
	for i in range(len(givens)):
		print('{:6}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}'.format(', '.join(givens[i]), P[i, 0], P[i, 1], P[i, 2], P[i, 3]))


if __name__ == "__main__":
	num_samples = 1000
	n = 3
	freq = {}
	print('Generating {} samples from CFG:'.format(num_samples))
	print(CFG_STRING)

	all_sentences = generate_all_possible_sentences()
	all_samples = []

	# generate sample
	for _ in range(num_samples):
		sample = all_sentences[np.random.randint(len(all_sentences))]
		all_samples.append(sample)

		# get n-grams and add to frequency counter
		n_grams = ngrams(sample.split(), n)
		for g in n_grams:
			if g in freq:
				freq[g] += 1
			else:
				freq[g] = 1

	# print frequencies
	print('Frequency Table:')
	for gram, count in freq.items():
		print('{} \t {}'.format(', '.join(gram), count))

	print(SEPARATOR)

	# Transition matrix
	givens = list(itertools.product(tokens, repeat=n-1))
	P = freq_to_probabilities(n, givens, freq)
	print('Transition Matrix:')
	print_transition_matrix(givens, P)

	print(SEPARATOR)

	# Example to show that trigram can't remember the opening bracket
	index_given = givens.index(('*', '1'))
	index_next = tokens.index(')')
	print('Sample input: ( 1 * 1 * 1 ) ')
	print('Last trigram: \' * 1 ) \'')
	print('Last trigram observed probability: p( \')\' | \'*1\' ) = {:.3f}'.format(P[index_given, index_next]))
	index_max = np.argmax(P[index_given,:])
	print('Last trigram most probable observation: argmax p( x | \'*1\' ) = {}  with probability {:.3f}'.format(tokens[index_max], P[index_given, index_max]))





