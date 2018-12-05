"""
Using Pomegranate library because hmmlearn doesn't do scoring properly and is slow
Hope pomegranate is better
"""
import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution
from pomegranate.callbacks import ModelCheckpoint
from analyze.db_runner import DBRunner

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


class RuleJavaTokenHMMTrain:
    def __init__(self):
        # train_data = np.load('train_data.npy', mmap_mode='r')
        # train_data = np.memmap('train_data.npy', dtype='object', mode='r+', shape=(2317031,))

        # Uncomment this to increase a new datafile with ~250 samples
        # train_data = np.load('train_data.npy')
        # temp = train_data[0: train_data.shape[0] // 10000]
        # del train_data
        # train_data = temp
        # print(train_data.shape)
        # np.save('new_data2.npy',train_data)


        # maxlen = max(len(r) for r in train_data)
        # print(maxlen)
        # X = np.full([len(train_data), maxlen], np.nan)
        # for enu, row in enumerate(train_data):
        #     X[enu, :len(row)] += row.astype(int) 


        X = np.load('new_data2.npy')

        print('Data loaded.')
        print(X.shape)
        # np.save('new_data.npy', X)
        # self.model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=100, X=X, batch_size=1000, verbose=True)
        self.model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=10, X=X, verbose=True, stop_threshold=1e-2)

        input_tokens = list(map(lambda x: x, TEST_SEQ.split()))
        for idx in range(1, len(input_tokens)):
            score = self.model.log_probability(input_tokens[:idx])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))
        print('Done.')

        # trans_mat = np.load("rule_trans.npy")
        # emissions = np.load("rule_em.npy")

        # # start only at the first state
        # starts = np.zeros(len(trans_mat))
        # starts[0] = 1

        # raw_dists = []
        # for emission in emissions:
        #     em = DiscreteDistribution(dict(enumerate(emission)))
        #     raw_dists.append(em)
        # self.model = HiddenMarkovModel.from_matrix(
        #     trans_mat, raw_dists, starts)


        # # Read stdout from tokenize_all and save
        # f = io.StringIO()
        # with redirect_stdout(f):
        #     DBRunner().tokenize_all_db_source(output_type="name")
        # out = f.getvalue()
        # TEST_SEQ_ARR = out.splitlines()

        # sc_parser = SourceCodeParser()

        # # Convert training input strings into list of observations for the fit method
        # split_string = TEST_SEQ_ARR[0].split()
        # input_tokens = list(map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], split_string))
        # X = [[i] for i in input_tokens]
        # lengths = [len(X)]

        # for line in TEST_SEQ_ARR[1:]:
        #     input_tokens = list(map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], line.split()))
        #     input_tokens = [[i] for i in input_tokens]
        #     X = np.concatenate([X, input_tokens])
        #     lengths.append(len(input_tokens))

        # X = LabelEncoder().fit_transform(np.ravel(X))
        # X = np.array([X]).T

        # # -----------------------------
        # # Same as RuleJavaTokenHMM, but use the mats as initial distributions and try to learn actuals using input data

        # # load atn matrices
        # rule_trans = np.load("rule_trans.npy")
        # rule_em = np.load("rule_em.npy")

        # # start only at the first state
        # start_probs = np.zeros(len(rule_trans))
        # start_probs[0] = 1

        # self.model = MultinomialHMM(
        #     n_components=len(rule_trans),
        #     startprob_prior=start_probs,
        #     transmat_prior=rule_trans,
        #     # n_iter=100
        #     # init_params="se",
        #     verbose=True
        # )
        # self.model.n_features = rule_em.shape[1]
        # self.model.transmat_ = rule_trans
        # self.model.startprob_ = start_probs
        # self.model.emissionprob_ = rule_em

        # self.model.fit(X, lengths)
        # print(self.model)


class TrainedJavaTokenHMM:
    def __init__(self, num_hidden_states):

        #train_mat = DBRunner().tokenize_all_db_source_gen(output_type="np_id")
        train_mat = np.load("train_data_size_1000.npy")

        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            num_hidden_states,
            train_mat,
            verbose=True,
            stop_threshold=1e-4,
            name="TrainedJavaTokenHMM",
            n_jobs=-1, # maximum parallelism
            callbacks=[ModelCheckpoint(verbose=True)]
        )

        print("EVAL TEST SEQUENCE")
        input_tokens = list(map(lambda x: x, TEST_SEQ.split()))
        for idx in range(1, len(input_tokens)):
            score = self.model.log_probability(input_tokens[:idx])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))
        print('Done.')
