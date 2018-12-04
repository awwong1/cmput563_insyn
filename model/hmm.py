"""
Baseline models for Hidden Markov Models with discrete emissions
"""
import numpy as np

from hmmlearn.hmm import MultinomialHMM
from analyze.parser import SourceCodeParser

TEST_SEQ = "34 2 73 2 73 2 71 27 2 73 2 73 2 71 27 2 73 2 73 2 71 37 11 2 19 2 77 2 76 67 37 2 65 66 67 42 65 2 73 11 66 71 68 68 1"


class ATNJavaTokenHMM:
    """
    Multinomial HMM for java token predictions.
    Emissions are discrete over number of Java token types
    """

    def __init__(self):
        # load atn matrices
        atn_trans = np.load("atn_trans.npy")
        atn_em = np.load("atn_em.npy")

        # start only at the first state
        start_probs = np.zeros(len(atn_trans))
        start_probs[0] = 1

        self.model = MultinomialHMM(
            n_components=len(atn_trans),
            startprob_prior=start_probs,
            transmat_prior=atn_trans,
            # init_params="se",
            verbose=True
        )
        self.model.n_features = atn_em.shape[1]
        self.model.transmat_ = atn_trans
        self.model.startprob_ = start_probs
        self.model.emissionprob_ = atn_em

        # print(self.model)
        # input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        # for idx in range(1, len(input_tokens)):
        #     score = self.model.score([input_tokens[:idx]])
        #     print("idx: {idx}/{total} score: {score}".format(idx=idx,
        #                                                      total=len(input_tokens)-1, score=score))

    def score(self, token_sequence_ids):
        return self.model.score([token_sequence_ids])

class RuleJavaTokenHMM:
    def __init__(self):
        # load atn matrices
        rule_trans = np.load("rule_trans.npy")
        rule_em = np.load("rule_em.npy")

        # start only at the first state
        start_probs = np.zeros(len(rule_trans))
        start_probs[0] = 1

        self.model = MultinomialHMM(
            n_components=len(rule_trans),
            startprob_prior=start_probs,
            transmat_prior=rule_trans,
            # init_params="se",
            verbose=True
        )
        self.model.n_features = rule_em.shape[1]
        self.model.transmat_ = rule_trans
        self.model.startprob_ = start_probs
        self.model.emissionprob_ = rule_em

        # print(self.model)
        # input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        # for idx in range(1, len(input_tokens)):
        #     score = self.model.score([input_tokens[:idx]])
        #     print("idx: {idx}/{total} score: {score}".format(idx=idx,
        #                                                      total=len(input_tokens)-1, score=score))
            # pred = self.model.predict(np.array([input_tokens[:idx]]).T)
            # hr_preds = list(map(lambda x: JavaParser.ruleNames[x], pred))
            # print(hr_preds)

    def score(self, token_sequence_ids):
        return self.model.score([token_sequence_ids])
