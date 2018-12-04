"""
Baseline models for Hidden Markov Models with discrete emissions
"""
import numpy as np

from hmmlearn.hmm import MultinomialHMM
from analyze.parser import SourceCodeParser
from analyze.db_runner import DBRunner
from sklearn.preprocessing import LabelEncoder
import io
from contextlib import redirect_stdout

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

        print(self.model)
        input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        for idx in range(1, len(input_tokens)):
            score = self.model.score([input_tokens[:idx]])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))


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

        print(self.model)
        input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        for idx in range(1, len(input_tokens)):
            score = self.model.score([input_tokens[:idx]])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))

        # score = self.model.score([input_tokens])
        # print(score)
        # value error when getting posteriors
        # print([input_tokens])
        score, posteriors = self.model.score_samples(np.array([input_tokens]).T)
        # score, posteriors = self.model.score_samples(input_tokens)
        print("Score:")
        print(score)
        print("Posteriors")
        print(posteriors)
        print(posteriors.shape)



class RuleJavaTokenHMMTrain:
    def __init__(self):

        # Read stdout from tokenize_all and save
        f = io.StringIO()
        with redirect_stdout(f):
            DBRunner().tokenize_all_db_source(output_type="name")
        out = f.getvalue()
        TEST_SEQ_ARR = out.splitlines()

        sc_parser = SourceCodeParser()

        # Convert training input strings into list of observations for the fit method
        split_string = TEST_SEQ_ARR[0].split()
        input_tokens = list(map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], split_string))
        X = [[i] for i in input_tokens]
        lengths = [len(X)]

        for line in TEST_SEQ_ARR[1:]:
            input_tokens = list(map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], line.split()))
            input_tokens = [[i] for i in input_tokens]
            X = np.concatenate([X, input_tokens])
            lengths.append(len(input_tokens))

        X = LabelEncoder().fit_transform(np.ravel(X))
        X = np.array([X]).T

        # -----------------------------
        # Same as RuleJavaTokenHMM, but use the mats as initial distributions and try to learn actuals using input data

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
            # n_iter=100
            # init_params="se",
            verbose=True
        )
        self.model.n_features = rule_em.shape[1]
        self.model.transmat_ = rule_trans
        self.model.startprob_ = start_probs
        self.model.emissionprob_ = rule_em

        self.model.fit(X, lengths)
        print(self.model)

        # TODO: Add epsilons and renormalize

        # input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        # for idx in range(1, len(input_tokens)):
        #     score = self.model.score([input_tokens[:idx]])
        #     print("idx: {idx}/{total} score: {score}".format(idx=idx,
        #                                                      total=len(input_tokens)-1, score=score))



