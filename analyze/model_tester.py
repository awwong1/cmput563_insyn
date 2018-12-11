"""
this whole file requires refactoring
"""
import logging
import fnmatch
import os
import random
import numpy as np
from multiprocessing import Pool, cpu_count, current_process
from analyze.parser import SourceCodeParser
from model.ngram import KenLM10Gram, KenLM2Gram
from model.hmm_pom import ATNJavaTokenHMM, RuleJavaTokenHMM, Trained10StateHMM, Trained100StateHMM, TrainedSmoothStateHMM


class ModelTester:
    """
    Take java source code, perform one token modification (add, remove, change),
    then use the ngram to suggest fix.
    """
    logger = logging.getLogger(__name__)
    n10_gram = None  # set in a class method call
    n2_gram = None
    atn_hmm = None
    rule_hmm = None
    t10_hmm = None
    t100_hmm = None
    tsmooth_hmm = None

    pattern = "*.java"
    all_token_types = list(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())
    all_token_type_ids = list(
        map(lambda x: str(x), SourceCodeParser.JAVA_TOKEN_TYPE_MAP.values()))
    try_num_locations = 10  # How many locations do we try?
    num_suggestions = 1000  # How many suggestions do we reveal?

    if "CONST" in all_token_types:
        idx = all_token_types.index("CONST")
        all_token_types.remove("CONST")
        del all_token_type_ids[idx]
    if "GOTO" in all_token_types:
        idx = all_token_types.index("GOTO")
        all_token_types.remove("GOTO")
        del all_token_type_ids[idx]

    @staticmethod
    def _get_first(tup):
        """multiprocessing fails if lambad function exists within the target
        """
        return tup[0]

    @staticmethod
    def _javac_init():
        """
        Called by pool initializer in tokenize_all_db_source.
        Setup SourceCodeParser javac instance (py4j JavaGateway binding)
        """
        global sc_parser
        sc_parser = SourceCodeParser()

    @staticmethod
    def _tokenize_file_contents(source_path):
        """
        async safe tokenize file contents
        """
        with open(source_path, "r") as source_file:
            source_code = source_file.read()
            (num_errors, tuple_tokens) = sc_parser.javac_analyze(source_code)
            if num_errors:
                ModelTester.logger.error("{path} contains {num_errs} error(s), skipping".format(
                    num_errs=num_errors, path=source_path),  # sc_parser.javac_check_syntax(source_code)
                )
            else:
                # return source_path, map(lambda x: x[0], tuple_tokens) # pickle fails with lambda
                return source_path, list(map(ModelTester._get_first, tuple_tokens))

    @staticmethod
    def _perform_one_token_break(source_and_tokens):
        """
        randomly perforn an ADD, DELETE or MODIFY on the original tokens sequence, put into test_tokens sequence
        """
        (source_path, original_tokens) = source_and_tokens  # unpack the dict item
        test_tokens = original_tokens.copy()

        change_type = random.choice(["ADD", "DEL", "MOD"])
        change_idx = random.randrange(0, len(original_tokens))
        ModelTester.logger.debug(" ".join(original_tokens))
        token = "ERROR"  # set in the 3 change types
        if change_type == "ADD":
            add_token_types = ModelTester.all_token_types.copy()
            if "CUSTOM" in add_token_types:
                add_token_types.remove("CUSTOM")
            if "EOF" in add_token_types:
                add_token_types.remove("EOF")

            # ---Jake Test
            if "CONST" in add_token_types:
                add_token_types.remove("CONST")
            if "GOTO" in add_token_types:
                add_token_types.remove("GOTO")
            # ---Jake Test

            rand_token = random.choice(add_token_types)
            test_tokens.insert(change_idx, rand_token)
            ModelTester.logger.info("{}: BREAK by {} {} at {}".format(source_path,
                                                                      change_type, rand_token, change_idx))
            token = rand_token
        elif change_type == "DEL":
            token = test_tokens.pop(change_idx)
            ModelTester.logger.info("{}: BREAK by {} {} at {}".format(source_path, change_type,
                                                                      original_tokens[change_idx], change_idx))
        elif change_type == "MOD":
            sub_token_types = ModelTester.all_token_types.copy()
            sub_token_types.remove(original_tokens[change_idx])
            if "CUSTOM" in sub_token_types:
                sub_token_types.remove("CUSTOM")
            if "EOF" in sub_token_types:
                sub_token_types.remove("EOF")

            # ---Jake Test
            if "CONST" in sub_token_types:
                sub_token_types.remove("CONST")
            if "GOTO" in sub_token_types:
                sub_token_types.remove("GOTO")
            # ---Jake Test

            rand_token = random.choice(sub_token_types)
            test_tokens[change_idx] = rand_token
            ModelTester.logger.info("{}: BREAK by {} from {} to {} at {}".format(
                source_path, change_type, original_tokens[change_idx], rand_token, change_idx
            ))
            token = rand_token
        return (source_path, original_tokens, test_tokens, change_type, change_idx, token)

    # ============================
    # NGRAM RELATED STATIC METHODS
    # ============================

    @staticmethod
    def _ngram_locate_and_fix(source_and_err_tokens, model_name="n/a"):

        if model_name == "n10_gram":
            ngram_model = ModelTester.n10_gram
        elif model_name == "n2_gram":
            ngram_model = ModelTester.n2_gram
        else:
            raise RuntimeError("Model not defined: {}".format(model_name))

        # LOCATE
        (source_path, test_tokens) = source_and_err_tokens  # unpack the dict item
        str_test_tokens = " ".join(test_tokens)
        counter = 0  # counter will go to full length of test_tokens due to </s>
        accum_score = 0
        token_idx_prob = []
        for prob, ngram_len, _ in ngram_model.full_scores(str_test_tokens):
            accum_score += prob
            ModelTester.logger.debug(
                "{path}: LOCATE_ERROR_SCORE {score} ({counter} of {total}) ngram={ngram_len} [{model_name}]".format(
                    path=source_path,
                    score=accum_score,
                    counter=counter,
                    total=len(test_tokens),
                    ngram_len=ngram_len,
                    model_name=model_name
                ))
            token_idx_prob.append((counter, prob,))
            counter += 1
        token_idx_prob.sort(key=lambda x: x[1])

        fix_prob = []
        # For the most likely error locations, try add, mod, and delete
        for token_idx, score in token_idx_prob[:ModelTester.try_num_locations]:
            ModelTester.logger.info(
                "{path}: CHECKING_LOCATION {token_idx} ({score}) [{model_name}]".format(
                    path=source_path,
                    token_idx=token_idx,
                    score=score,
                    model_name=model_name
                ))
            if token_idx == len(test_tokens):
                # edge case, we cannot modify or delete </s>, only add to end of sequence
                for add_token in ModelTester.all_token_types:
                    fix_tokens_by_add = test_tokens.copy()
                    fix_tokens_by_add.append(add_token)
                    str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                    new_score_by_add = ngram_model.score(
                        str_fix_by_add_tokens)
                    fix_prob.append(
                        (new_score_by_add, fix_tokens_by_add, "ADD", token_idx, add_token))
                continue

            to_change_token = test_tokens[token_idx]
            # try adding token
            for add_token in ModelTester.all_token_types:
                fix_tokens_by_add = test_tokens.copy()
                fix_tokens_by_add.insert(token_idx, add_token)
                str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                new_score_by_add = ngram_model.score(
                    str_fix_by_add_tokens)
                fix_prob.append(
                    (new_score_by_add, fix_tokens_by_add, "ADD", token_idx, add_token))

            # try changing token, cannot mod into itself
            sub_token_types = ModelTester.all_token_types.copy()
            sub_token_types.remove(to_change_token)
            for mod_token in sub_token_types:
                fix_tokens_by_mod = test_tokens.copy()
                fix_tokens_by_mod[token_idx] = mod_token
                str_fix_by_mod_tokens = " ".join(fix_tokens_by_mod)
                new_score_by_mod = ngram_model.score(
                    str_fix_by_mod_tokens)
                fix_prob.append(
                    (new_score_by_mod, fix_tokens_by_mod, "MOD", token_idx, mod_token))

            # try deleting token
            fix_tokens_by_del = test_tokens.copy()
            fix_tokens_by_del.pop(token_idx)
            str_fix_by_del_tokens = " ".join(fix_tokens_by_del)
            new_score_by_del = ngram_model.score(str_fix_by_del_tokens)
            fix_prob.append(
                (new_score_by_del, fix_tokens_by_del, "DEL", token_idx, to_change_token))

        fix_prob.sort(key=lambda x: x[0], reverse=True)
        return (source_path, fix_prob)

    @staticmethod
    def _10_gram_locate_and_fix(source_and_err_tokens):
        return ModelTester._ngram_locate_and_fix(source_and_err_tokens, "n10_gram")

    @staticmethod
    def _2_gram_locate_and_fix(source_and_err_tokens):
        return ModelTester._ngram_locate_and_fix(source_and_err_tokens, "n2_gram")


    # ============================
    # HMM RELATED STATIC METHODS
    # ============================

    @staticmethod
    def _atn_score(seq_idx_and_test_seq):
        seq_idx, test_seq = seq_idx_and_test_seq
        score = ModelTester.atn_hmm.score(test_seq)
        return (score, seq_idx)

    @staticmethod
    def _rule_score(seq_idx_and_test_seq):
        seq_idx, test_seq = seq_idx_and_test_seq
        score = ModelTester.rule_hmm.score(test_seq)
        return (score, seq_idx)

    @staticmethod
    def _train10_score(seq_idx_and_test_seq):
        seq_idx, test_seq = seq_idx_and_test_seq
        score = ModelTester.t10_hmm.score(test_seq)
        return (score, seq_idx)

    @staticmethod
    def _train100_score(seq_idx_and_test_seq):
        seq_idx, test_seq = seq_idx_and_test_seq
        score = ModelTester.t100_hmm.score(test_seq)
        return (score, seq_idx)

    @staticmethod
    def _trainsmooth_score(seq_idx_and_test_seq):
        seq_idx, test_seq = seq_idx_and_test_seq
        score = ModelTester.tsmooth_hmm.score(test_seq)
        return (score, seq_idx)

    @staticmethod
    def _hmm_locate_and_fix(source_and_err_tokens, model_name="n/a"):

        if model_name == "atn-hmm":
            SCORE_FUNC = ModelTester._atn_score
        elif model_name == "rule-hmm":
            SCORE_FUNC = ModelTester._rule_score
        elif model_name == "t10-hmm":
            SCORE_FUNC = ModelTester._train10_score
        elif model_name == "t100-hmm":
            SCORE_FUNC = ModelTester._train100_score
        elif model_name == "tsmooth-hmm":
            SCORE_FUNC = ModelTester._trainsmooth_score
        else:
            raise RuntimeError("Model not defined: {}".format(model_name))

        # LOCATE THE MOST LIKELY ERROR LOCATIONS USING THE HMM
        (source_path, test_tokens) = source_and_err_tokens  # unpack the dict item
        test_seq_ids = list(
            map(lambda x: str(SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x]), test_tokens))
        token_idx_prob = []
        _locate_pre_add_fix = dict()
        _locate_add_fix_score = dict()
        for seq_idx in range(1, len(test_seq_ids)):
            _locate_pre_add_fix[seq_idx] = test_seq_ids[:seq_idx]

        for (score, add_fix_seq_idx) in map(SCORE_FUNC, _locate_pre_add_fix.items()):
            _locate_add_fix_score[add_fix_seq_idx] = score
            ModelTester.logger.info(
                "{path}: SCORING_LOCATION {token_idx} ({score}) [{name}]".format(
                    name=model_name,
                    path=source_path,
                    token_idx=add_fix_seq_idx,
                    score=score
                ))
        cur_score = 0
        for seq_idx in range(1, len(test_seq_ids)):
            seq_score = _locate_add_fix_score[seq_idx]
            contrib_score = seq_score - cur_score
            cur_score = seq_score
            ModelTester.logger.info(
                "{path}: CHECKING_LOCATION_SCORE {token_idx} ({score}) [{name}]".format(
                    name=model_name,
                    path=source_path,
                    token_idx=seq_idx,
                    score=contrib_score
                ))
            token_idx_prob.append((seq_idx, contrib_score))

        # normalize
        # mean = np.mean([i[1] for i in token_idx_prob])
        # stdev = np.std([i[1] for i in token_idx_prob])
        # temp = []
        # for i in token_idx_prob:
        #     temp.append((seq_idx, (i[1] - mean) / stdev))
        # token_idx_prob = temp

        token_idx_prob.sort(key=lambda x: x[1])

        # RECCOMEND LIKELY FIXES USING THE HMM
        fix_prob = []

        # do all adds for zero in case break is in the first element...
        _zero_add_fix = dict()
        for add_token in ModelTester.all_token_type_ids:
            fix_tokens_by_add = test_seq_ids.copy()
            fix_tokens_by_add.insert(0, add_token)
            _zero_add_fix[add_token] = fix_tokens_by_add
        for (score, add_token) in map(SCORE_FUNC, _zero_add_fix.items()):
            ModelTester.logger.debug(
                "{path}: CHECK_ADD {token_idx} BEFORE SEQUENCE ({score}) [{name}]".format(
                    name=model_name,
                    path=source_path,
                    token_idx=add_token,
                    score=score
                ))
            fix_prob.append(
                (score, _zero_add_fix[add_token], "ADD", 0, add_token))

        # For the most likely error locations, try add, mod, and delete
        for token_idx, score in token_idx_prob[:ModelTester.try_num_locations]:
            to_change_token = test_seq_ids[token_idx]
            ModelTester.logger.info(
                "{path}: CHECKING_LOCATION {token_idx} ({score}) [{name}]".format(
                    name=model_name,
                    path=source_path,
                    token_idx=token_idx,
                    score=score
                ))
            # try adding token
            _add_fix = dict()
            for add_token in ModelTester.all_token_type_ids:
                fix_tokens_by_add = test_seq_ids.copy()
                fix_tokens_by_add.insert(token_idx, add_token)
                _add_fix[add_token] = fix_tokens_by_add
            for (score, add_token) in map(SCORE_FUNC, _add_fix.items()):
                ModelTester.logger.debug(
                    "{path}: CHECKING_ADD {token} AT {pos} ({score}) [{name}]".format(
                        name=model_name,
                        path=source_path,
                        token=add_token,
                        pos=token_idx,
                        score=score
                    ))
                fix_prob.append(
                    (score, _add_fix[add_token], "ADD", token_idx, add_token))

            # try changing token, cannot mod into itself
            sub_token_types = ModelTester.all_token_type_ids.copy()
            sub_token_types.remove(to_change_token)
            _sub_fix = dict()
            for mod_token in sub_token_types:
                fix_tokens_by_mod = test_seq_ids.copy()
                fix_tokens_by_mod[token_idx] = mod_token
                _sub_fix[mod_token] = fix_tokens_by_mod
            for (score, mod_token) in map(SCORE_FUNC, _sub_fix.items()):
                ModelTester.logger.debug(
                    "{path}: CHECKING_MOD {token} AT {pos} ({score}) [{name}]".format(
                        name=model_name,
                        path=source_path,
                        token=mod_token,
                        pos=token_idx,
                        score=score
                    ))
                fix_prob.append(
                    (score, _sub_fix[mod_token], "MOD", token_idx, mod_token))

            # try deleting token
            fix_tokens_by_del = test_seq_ids.copy()
            del_token = fix_tokens_by_del.pop(token_idx)
            (new_score_by_del, _) = SCORE_FUNC([del_token, fix_tokens_by_del])
            ModelTester.logger.debug(
                "{path}: CHECKING_DEL {token} AT {pos} ({score}) [{name}]".format(
                    name=model_name,
                    path=source_path,
                    token=del_token,
                    pos=token_idx,
                    score=new_score_by_del
                ))
            fix_prob.append(
                (new_score_by_del, fix_tokens_by_del, "DEL", token_idx, to_change_token))

        fix_prob.sort(key=lambda x: x[0], reverse=True)
        return (source_path, fix_prob)

    @staticmethod
    def _rule_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="rule-hmm")

    @staticmethod
    def _atn_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="atn-hmm")

    @staticmethod
    def _train10_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="t10-hmm")

    @staticmethod
    def _train100_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="t100-hmm")

    @staticmethod
    def _trainsmooth_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="tsmooth-hmm")

    @staticmethod
    def _mean_reciprocal_rank(ranks):
        """
        Take all ranks, calculate meanreciprocal rank.
        Ranks are integers ranging from 1 to {num_suggestions}
        """
        reciprocal_ranks = map(lambda x: 1/x, ranks)
        return sum(reciprocal_ranks)/len(ranks)

    @staticmethod
    def _print_model_summary(rank_list, model_name="n/a"):
        mrr = ModelTester._mean_reciprocal_rank(rank_list)
        model_found = [x for x in rank_list if x < ModelTester.num_suggestions]
        print("{name} found {found}/{total} true fixes (Mean Reciprocal Rank={mrr:.2})".format(
            name=model_name,
            found=len(model_found), total=len(rank_list), mrr=mrr))
        print(rank_list)

    def __init__(self, input_file_path):
        """
        Get all the .java files in the user provided path
        """
        self.file_to_tokens = dict()  # original token sequence
        self.file_paths = []

        # get list of all source files to test
        self.file_paths = []
        if os.path.isfile(input_file_path):
            self.file_paths = fnmatch.filter(
                [input_file_path], self.pattern)
        elif os.path.isdir(input_file_path):
            for dirpath, _dirnames, filenames in os.walk(input_file_path):
                if not filenames:
                    continue
                match_files = fnmatch.filter(filenames, self.pattern)
                for match_file in match_files:
                    self.file_paths.append(
                        os.path.join(dirpath, match_file))

        if not len(self.file_paths):
            raise FileNotFoundError("No valid {pattern} files found for {path}".format(
                pattern=self.pattern, path=input_file_path))

    def _pre_evaluation(self):
        with Pool(initializer=ModelTester._javac_init) as pool:
            for path_and_tokens in pool.imap(ModelTester._tokenize_file_contents, self.file_paths):
                if path_and_tokens:
                    (source_path, tokens_type) = path_and_tokens
                    if tokens_type:
                        self.file_to_tokens[source_path] = list(tokens_type)

        if not len(self.file_to_tokens.keys()):
            raise ValueError("All {pattern} files syntactically incorrect".format(
                pattern=self.pattern))

    def _run_model_evaluation(self, loc_and_fix_func, model_name="n/a"):
        model_ranks = []
        with Pool() as pool:
            for (source_path, fix_probs) in pool.imap(loc_and_fix_func, self.one_error_tokens.items()):
                orig_seq = self.file_to_tokens[source_path]
                org_ints = list(map(lambda x: str(SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x]), self.file_to_tokens[source_path]))
                rank = 1
                for fix_prob in fix_probs:
                    # unpack
                    new_score, fix_sequence, action, token_idx, to_change_token = fix_prob
                    ModelTester.logger.info("%s: RANK %d SUGGEST %s %s AT %d (score: %.2f)",
                                            model_name, rank, action,
                                            to_change_token, token_idx, new_score)
                    if fix_sequence == orig_seq or fix_sequence == org_ints:
                        break
                    else:
                        rank += 1
                model_ranks.append(rank)
        return model_ranks

    def run_evaluation(self):
        self._pre_evaluation()

        # PERFORM ONE TOKEN BREAKS FOR ALL SYNTACTICALLY VALID SOURCE CODE
        self.one_error_tokens = dict()
        self.eval_str = dict()
        with Pool() as pool:
            for (source_path, _, test_tokens, change_type, change_idx, change_token) in pool.imap(ModelTester._perform_one_token_break, self.file_to_tokens.items()):
                eval_str = "{change} {c_token} at {t_idx}".format(
                    change=change_type, c_token=change_token, t_idx=change_idx
                )
                self.one_error_tokens[source_path] = test_tokens
                self.eval_str[source_path] = eval_str

        # PERFORM PROBABALISTIC LOCATION DETECTION AND FIX RECCOMENDATION
        n10gram_ranks = self._run_model_evaluation(ModelTester._10_gram_locate_and_fix, "n10_gram")
        n2gram_ranks = self._run_model_evaluation(ModelTester._2_gram_locate_and_fix, "n2_gram")
        # rule_ranks = self._run_model_evaluation(ModelTester._rule_hmm_locate_and_fix, "rule-hmm")
        # atn_ranks = self._run_model_evaluation(ModelTester._atn_hmm_locate_and_fix, "atn-hmm")
        # t10_ranks = self._run_model_evaluation(ModelTester._train10_hmm_locate_and_fix, "t10-hmm")
        # t100_ranks = self._run_model_evaluation(ModelTester._train100_hmm_locate_and_fix, "t100-hmm")
        tsmooth_ranks = self._run_model_evaluation(ModelTester._trainsmooth_hmm_locate_and_fix, "tsmooth-hmm")

        # PRINT SUMMARY OF RESULTS AND MODEL PERFORMANCE
        print("\n---- SUMMARY OF CHANGES ----")
        for source_path, eval_str in self.eval_str.items():
            print("tokenized {s_path}, performed {eval_str}".format(
                s_path=source_path,
                eval_str=eval_str
            ))
        print("\n---- MODEL PERFORMANCE ----")
        ModelTester._print_model_summary(n10gram_ranks, model_name="n10_gram")
        ModelTester._print_model_summary(n2gram_ranks, model_name="n2_gram")
        # ModelTester._print_model_summary(rule_ranks, model_name="rule-hmm")
        # ModelTester._print_model_summary(atn_ranks, model_name="atn-hmm")
        # ModelTester._print_model_summary(t10_ranks, model_name="t10-hmm")
        # ModelTester._print_model_summary(t100_ranks, model_name="t100-hmm")
        ModelTester._print_model_summary(tsmooth_ranks, model_name="tsmooth-hmm")
        print()

    @classmethod
    def init_models(cls):
        cls.n10_gram = KenLM10Gram()
        cls.n2_gram = KenLM2Gram()
        # cls.atn_hmm = ATNJavaTokenHMM()
        # cls.rule_hmm = RuleJavaTokenHMM()
        # cls.t10_hmm = Trained10StateHMM()
        # cls.t100_hmm = Trained100StateHMM()
        cls.tsmooth_hmm = TrainedSmoothStateHMM()

