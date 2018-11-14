#!/usr/bin/env python3

from nltk.parse.generate import generate
from nltk import CFG, ngrams
import numpy as np
import itertools
from hmmlearn import hmm
from colorama import init

init()

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


def train_ngram(n, num_samples):
	freq = {}
	all_sentences = generate_all_possible_sentences()
	all_samples = []

	# generate sample
	for _ in range(num_samples):
		sample = all_sentences[np.random.randint(len(all_sentences))]
		all_samples.append( [[tokens.index(token)] for token in sample.split()] )
		# all_samples.append(sample.split())

		# get n-grams and add to frequency counter
		n_grams = ngrams(sample.split(), n)
		for g in n_grams:
			if g in freq:
				freq[g] += 1
			else:
				freq[g] = 1

	return freq, all_samples


def _create_incorrect_sample(sample):
	sample_arr = sample.split()
	index_modified = np.random.randint(2, len(sample_arr))
	tokens_to_choose_from = [tokens[i] for i in range(len(tokens)) if sample_arr[index_modified] != tokens[i]]
	new_token = tokens_to_choose_from[np.random.randint(len(tokens_to_choose_from))]
	sample_arr[index_modified] = new_token
	return ' '.join(sample_arr), index_modified * 2


def _find_probable_correction(index_given, givens, wrong_token, P):
	# Set current probability and token as probability of seeing the mistake
	correction_p = P[index_given, tokens.index(wrong_token)]
	correction_t = wrong_token

	# Go through each possible token and see if its more probable given the previous tokens
	for i in range(len(tokens)):
		new_p = P[index_given, i]
		if new_p > correction_p:
			correction_p = new_p
			correction_t = tokens[i]

	return correction_p, correction_t





def test_ngram(num_test, givens, P):
	all_sentences = generate_all_possible_sentences()
	test_samples = []
	count = 0.0

	for i in range(num_test):

		# Get a sample that is at least 3 tokens (or else 3-gram model fails)
		sample = all_sentences[np.random.randint(len(all_sentences))]
		while(len(sample) < 3):
			sample = all_sentences[np.random.randint(len(all_sentences))]

		print("Test {}: Original sample: {}".format(i+1, sample))
		
		# Modify original input to create an error
		incorrect_sample, index_modified = _create_incorrect_sample(sample)
		print("Test {}: Modified sample: {}\x1b[6;30;42m{}\x1b[0m{}".format(i+1, incorrect_sample[0 : index_modified], incorrect_sample[index_modified], incorrect_sample[index_modified+1 : ]))

		# Probability of observing error
		index_given = givens.index((incorrect_sample[index_modified - 4], incorrect_sample[index_modified - 2]))
		index_next = tokens.index(incorrect_sample[index_modified])
		P[index_given, index_next]
		print('Test {}: Probability of observing error: {:.3f}'.format(i+1, P[index_given, index_next]))

		# Find most probable change
		correction_p, correction_t = _find_probable_correction(index_given, givens, incorrect_sample[index_modified], P)
		print('Test {}: Most probable fix \'{}\': {:.3f}'.format(i+1, correction_t, correction_p))

		# Check if correct recommendation
		if sample[index_modified] == correction_t:
			print('Test {}: Correct fix found!'.format(i+1))
			count = count + 1
		else:
			print('Test {}: Incorrect fix.'.format(i+1))

		test_samples.append([sample, incorrect_sample, index_modified])

	print('Test Accuracy: {:.2%}'.format(count / num_test))

	return test_samples



def test_hmm(test_samples, startprob, transmat, emissionprob, model):
	count = 0
	count2 = 0

	for i, test_case in enumerate(test_samples):
		# Get the same sample performed in n-gram
		sample = test_case[0]
		incorrect_sample = test_case[1]
		index_modified = test_case[2]

		# Manually finding n-step transition, hmm does this for us alread
		# start_state = np.random.choice(len(startprob), p=startprob)
		# # n_step_transmat = np.linalg.matrix_power(transmat, index_modified // 2)
		# n_step_transmat = transmat
		# print(start_state)
		# print(n_step_transmat)

		# # p(token_i) = sum_k p(token_i | z_i = k) * p(z_i = k)
		# p_tokens = []
		# for i in range(len(tokens)):
		# 	marginalization = 0
		# 	for k in range(len(n_step_transmat[start_state, :])):
		# 		marginalization = marginalization + (emissionprob[k, i] * n_step_transmat[start_state, k])

		# 	p_tokens.append(marginalization)

		# print(p_tokens)
		# index_most_prob = p_tokens.index(max(p_tokens))
		# print(tokens[index_most_prob])



		# Input the sequence of tokens before the mistake into the model
		# Then make a prediction for the next observed token
		state_sequence = model.predict( [[tokens.index(token)] for token in (sample[0:index_modified]).split()]  )
		p_tokens = model.transmat_[state_sequence[-1], :]
		index_most_prob = np.argmax(p_tokens)
		
		# Used to check the top 2 recommendations
		two_most_prob = sorted(range(len(p_tokens)), key=lambda x: p_tokens[x])[-2:]



		print("Test {}: Original sample: {}".format(i+1, sample))
		print("Test {}: Modified sample: {}\x1b[6;30;42m{}\x1b[0m{}".format(i+1, incorrect_sample[0 : index_modified], incorrect_sample[index_modified], incorrect_sample[index_modified+1 : ]))
		print('Test {}: Most probable fix \'{}\': {:.3f}'.format(i+1, tokens[index_most_prob], p_tokens[index_most_prob]))

		# Check if top recommendation given is correct
		if sample[index_modified] == tokens[index_most_prob]:
			print('Test {}: Correct fix found!'.format(i+1))
			count = count + 1

		# Check the top 2 recommendations
		if sample[index_modified] == tokens[two_most_prob[0]] or sample[index_modified] == tokens[two_most_prob[1]]:
			print('Test {}: Correct fix found!'.format(i+1))
			count2 = count2 + 1


	print('Test Accuracy First Recommendation: {:.2%}'.format(count / num_test))
	print('Test Accuracy First or Second Recommendation: {:.2%}'.format(count2 / num_test))






if __name__ == "__main__":
	num_samples = 1000
	num_test = 50
	n = 3
	print('Generating {} samples from CFG:'.format(num_samples))
	print(CFG_STRING)

	freq, all_samples = train_ngram(n, num_samples)

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

	# Test the n-gram model by making single-mistakes in valid samples, and finding most probable fix
	test_samples = test_ngram(num_test, givens, P)

	print(SEPARATOR)


	# ------------ HMM ---------------
	# Format input data
	train_data = np.concatenate(all_samples)
	lengths = [len(i) for i in all_samples]

	# Create HMM Model
	model = hmm.MultinomialHMM(n_components=3, n_iter=50).fit(train_data)
	hidden_states = model.predict(train_data)

	# Pretty print
	np.set_printoptions(precision=4)
	np.set_printoptions(suppress=True)

	# Get model attributes
	startprob = model.startprob_
	transmat = model.transmat_
	emissionprob = model.emissionprob_

	print("Initial State Probabilities:")
	print(startprob)
	print("Transition Matrix:")
	print(transmat)
	print("Emission Probabilities")
	print(emissionprob)
	print("Score:")
	print(model.score(train_data))

	# Test the HMM model using same samples tested in n-gram
	test_hmm(test_samples, startprob, transmat, emissionprob, model)














