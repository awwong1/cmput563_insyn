"""
Using Pomegranate library because hmmlearn doesn't do scoring properly and is slow
Hope pomegranate is better
"""
import os
import tables
import numpy as np
import logging
import json
from pomegranate import HiddenMarkovModel, DiscreteDistribution, State
from pomegranate.callbacks import ModelCheckpoint
from analyze.db_runner import DBRunner

from analyze.parser import SourceCodeParser
from grammar.JavaParser import JavaParser

TEST_SEQ = "33 1 72 1 72 1 70 26 1 72 1 72 1 70 26 1 72 1 72 1 70 36 10 1 18 1 76 1 75 66 36 1 64 65 66 41 64 1 72 10 65 70 67 67 0"

TRAINED_TRANS_MAT = "trained_trans_mat.npy"

class ATNJavaTokenHMM:
    """
    Multinomial HMM for java token predictions.
    Emissions are discrete over number of Java token types
    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.info("start initializing ATN-HMM")
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
        self.logger.info("done initializing ATN-HMM")

    def score(self, token_sequence_ids):
        # trained as int [0, 1, 2, ... 111]
        casted_seq = list(map(lambda x: int(x), token_sequence_ids))
        return self.model.log_probability(casted_seq)


class RuleJavaTokenHMM:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.info("start initializing rule-HMM")
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
        self.logger.info("done initializing rule-HMM")

    def score(self, token_sequence_ids):
        # trained as int [0, 1, 2, ... 111]
        casted_seq = list(map(lambda x: int(x), token_sequence_ids))
        return self.model.log_probability(casted_seq)


class Trained10StateHMM:
    logger = logging.getLogger(__name__)
    FILENAME = "TrainedJavaTokenHMM_10.1.json"

    def __init__(self):
        self.logger.info("start initializing t10-HMM")
        model_as_json = ""
        path_to_file = os.path.join(os.path.dirname(__file__), self.FILENAME)
        with open(path_to_file, "r") as f:
            model_as_json = f.read()
        self.model = HiddenMarkovModel.from_json(model_as_json)
        self.logger.info("done initializing t10-HMM")

    def score(self, token_sequence_ids):
        # trained as str ["0", "1", "2", ... "111"]
        casted_seq = list(map(lambda x: str(x), token_sequence_ids))
        return self.model.log_probability(casted_seq)


class Trained100StateHMM:
    logger = logging.getLogger(__name__)
    FILENAME = "TrainedJavaTokenHMM_100.1.json"

    def __init__(self):
        self.logger.info("start initializing t100-HMM")
        model_as_json = ""
        path_to_file = os.path.join(os.path.dirname(__file__), self.FILENAME)
        with open(path_to_file, "r") as f:
            model_as_json = f.read()
        self.model = HiddenMarkovModel.from_json(model_as_json)
        self.logger.info("done initializing t100-HMM")

    def score(self, token_sequence_ids):
        # trained as str ["0", "1", "2", ... "111"]
        casted_seq = list(map(lambda x: str(x), token_sequence_ids))
        return self.model.log_probability(casted_seq)



class TrainedSmoothStateHMM:
    logger = logging.getLogger(__name__)
    FILENAME = "LabelTrainedJaveTokenHMM.1.json"


    def __init__(self):
        self.logger.info("start initializing tsmooth-HMM")
        # model_as_json = ""
        # path_to_file = os.path.join(os.path.dirname(__file__), self.FILENAME)
        # with open(path_to_file, "r") as f:
        #     model_as_json = f.read()
        # self.model = HiddenMarkovModel.from_json(model_as_json)

        # Read back as json
        temp_mat = np.load(TRAINED_TRANS_MAT)
        distributions, names, start_dist = _get_model_json_data(self.FILENAME)
        self.model = HiddenMarkovModel.from_matrix(
            temp_mat,
            distributions,
            start_dist,
        )
        self.logger.info("done initializing tsmooth-HMM")


    def score(self, token_sequence_ids):
        # trained as str ["0", "1", "2", ... "111"]
        casted_seq = list(map(lambda x: int(x), token_sequence_ids))
        return self.model.log_probability(casted_seq)


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
        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution, n_components=10, X=X, verbose=True, stop_threshold=1e-2)

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
        table_file = tables.open_file("train_data_size_3000000.h5", mode='r')
        train_mat = table_file.root.data

        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            num_hidden_states,
            train_mat,
            verbose=True,
            algorithm="viterbi",
            stop_threshold=1e-4,
            name="TrainedJavaTokenHMM",
            n_jobs=-1,  # maximum parallelism
            callbacks=[ModelCheckpoint(verbose=True)],
        )

        print("EVAL TEST SEQUENCE")
        input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        print(TEST_SEQ)
        for idx in range(1, len(input_tokens)):
            score = self.model.log_probability(input_tokens[:idx])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))
            print("predicted states:")
            print(self.model.predict(input_tokens[:idx]))
            print()
        print('Done.')


def _get_model_json_data(json_name):
    with open(json_name, 'r') as f:
        model_json = json.load(f)

    distributions = []
    names = []
    model_states = model_json['states']
    for model_state in model_states:
        names.append(model_state['name'])
        state_dist = model_state['distribution']
        if state_dist is not None:
            params = state_dist['parameters'][0]
            dist_dict = {}
            for k,v in params.items():
                dist_dict[int(k)] = float(v)

        distributions.append(DiscreteDistribution(dist_dict))
    
    start_index = int(model_json["start_index"])
    start_dist = [1 if int(state) == start_index else 0 for state in names[:-2]]

    return distributions, names, start_dist



class LabelTrainedJavaTokenHMM:
    def __init__(self, num_hidden_states):
        # _get_model_json_data()
        tokens_file = tables.open_file("token_sequences.h5", mode='r')
        tokens_mat = tokens_file.root.data
        rules_file = tables.open_file("rule_sequences.h5", mode="r")
        rules_mat = rules_file.root.data

        print("Data Read")
        tokens_mat = tokens_mat[0:500000]
        rules_mat = rules_mat[0:500000]

        print(tokens_mat[0])
        print(rules_mat[0])

        print("State names")
        rules_list = []
        for i in rules_mat:
            rules_list = list(set(rules_list + i.tolist()))
        rules_list = list(set(rules_list))

        print("Rule formatting")
        temp = []
        for i in rules_mat:
            temp2 = []
            for j in i:
                temp2.append(str(j))
            temp.append(temp2)

        rules_mat = temp

        print("To NP")
        tokens_mat = np.array(tokens_mat)
        # rules_mat = np.array(rules_mat)


        print("Training")

        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            len(rules_list),
            # num_hidden_states,
            tokens_mat,
            verbose=True,
            algorithm="labeled",
            labels=rules_mat,
            # labels=temp,
            state_names=[str(i) for i in rules_list],
            pseudocount=10,
            use_pseudocount=True,
            stop_threshold=1e-4,
            name="LabelTrainedJaveTokenHMM",
            batch_size=1000,
            n_jobs=-1,  # maximum parallelism
            callbacks=[ModelCheckpoint(verbose=True)],
        )

        # Save data
        np.save(TRAINED_TRANS_MAT, self.model.dense_transition_matrix())


        print("EVAL TEST SEQUENCE")
        input_tokens = list(map(lambda x: int(x), TEST_SEQ.split()))
        print(TEST_SEQ)
        for idx in range(1, len(input_tokens)):
            score = self.model.log_probability(input_tokens[:idx])
            print("idx: {idx}/{total} score: {score}".format(idx=idx,
                                                             total=len(input_tokens)-1, score=score))
            print("predicted states:")
            print(self.model.predict(input_tokens[:idx]))
            print()
        print('Done.')

