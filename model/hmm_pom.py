"""
Using Pomegranate library because hmmlearn doesn't do scoring properly and is slow
Hope pomegranate is better
"""
import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution

from analyze.parser import SourceCodeParser

TEST_SEQ = "33 1 72 1 72 1 70 26 1 72 1 72 1 70 26 1 72 1 72 1 70 36 10 1 18 1 76 1 75 66 36 1 64 65 66 41 64 1 72 10 65 70 67 67 0"


class ATNJavaTokenHMM:
    """
    Multinomial HMM for java token predictions.
    Emissions are discrete over number of Java token types
    """

    def __init__(self):
        # load atn matrices
        trans_mat = np.load("atn_trans.npy")
        emissions = np.load("atn_em.npy")

        # start only at the first state
        starts = np.zeros(len(trans_mat))
        starts[0] = 1

        raw_dists = []
        for emission in emissions:
            em = DiscreteDistribution(dict(enumerate(emission)))
            raw_dists.append(em)


        self.model = HiddenMarkovModel.from_matrix(
            trans_mat, raw_dists, starts
        )

        # print(self.model)
        # input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        # for idx in range(1, len(input_tokens)):
        #     score = self.model.log_probability(input_tokens[:idx])
        #     print("idx: {idx}/{total} score: {score}".format(idx=idx,
        #                                                      total=len(input_tokens)-1, score=score))

    def score(self, token_sequence_ids):
        return self.model.log_probability(token_sequence_ids)


class RuleJavaTokenHMM:
    def __init__(self):
        # load atn matrices
        trans_mat = np.load("rule_trans.npy")
        emissions = np.load("rule_em.npy")

        # start only at the first state
        starts = np.zeros(len(trans_mat))
        starts[0] = 1

        raw_dists = []
        for emission in emissions:
            em = DiscreteDistribution(dict(enumerate(emission)))
            raw_dists.append(em)
        self.model = HiddenMarkovModel.from_matrix(
            trans_mat, raw_dists, starts)

        # print(self.model)
        # input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        # for idx in range(1, len(input_tokens)):
        #     score = self.model.log_probability(input_tokens[:idx])
        #     print("idx: {idx}/{total} score: {score}".format(idx=idx,
        #                                                      total=len(input_tokens)-1, score=score))

    def score(self, token_sequence_ids):
        return self.model.log_probability(token_sequence_ids)
