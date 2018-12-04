import logging
import fnmatch
import os
import random
from multiprocessing import Pool, cpu_count, current_process
from analyze.parser import SourceCodeParser
from model.ngram import KenLM10Gram
from model.hmm_pom import ATNJavaTokenHMM, RuleJavaTokenHMM


class ModelTester:
    """
    Take java source code, perform one token modification (add, remove, change),
    then use the ngram to suggest fix.
    """
    logger = logging.getLogger(__name__)
    ngram = None  # set in a class method call
    atn_hmm = None
    rule_hmm = None

    pattern = "*.java"
    all_token_types = list(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())
    all_token_type_ids = list(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.values())
    try_num_locations = 10  # How many locations do we try?
    num_suggestions = 1000  # How many suggestions do we reveal?

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
            add_token_types.remove("CUSTOM")
            add_token_types.remove("EOF")
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
            sub_token_types.remove("CUSTOM")
            sub_token_types.remove("EOF")
            rand_token = random.choice(sub_token_types)
            test_tokens[change_idx] = rand_token
            ModelTester.logger.info("{}: BREAK by {} from {} to {} at {}".format(
                source_path, change_type, original_tokens[change_idx], rand_token, change_idx
            ))
            token = rand_token
        return (source_path, original_tokens, test_tokens, change_type, change_idx, token)

    @staticmethod
    def _ngram_locate_and_fix(source_and_err_tokens):
        # LOCATE
        (source_path, test_tokens) = source_and_err_tokens  # unpack the dict item
        str_test_tokens = " ".join(test_tokens)
        counter = 0  # counter will go to full length of test_tokens due to </s>
        accum_score = 0

        token_idx_prob = []

        for prob, ngram_len, _ in ModelTester.ngram.full_scores(str_test_tokens):
            accum_score += prob
            ModelTester.logger.debug(
                "{path}: LOCATE_ERROR_SCORE {score} ({counter} of {total}) ngram={ngram_len}".format(
                    path=source_path,
                    score=accum_score,
                    counter=counter,
                    total=len(test_tokens),
                    ngram_len=ngram_len
                ))
            token_idx_prob.append((counter, prob,))
            counter += 1
        token_idx_prob.sort(key=lambda x: x[1])

        fix_prob = []
        # For the most likely error locations, try add, mod, and delete
        for token_idx, score in token_idx_prob[:ModelTester.try_num_locations]:
            ModelTester.logger.info(
                "{path}: CHECKING_LOCATION {token_idx} ({score})".format(
                    path=source_path,
                    token_idx=token_idx,
                    score=score
                ))
            if token_idx == len(test_tokens):
                # edge case, we cannot modify or delete </s>, only add to end of sequence
                for add_token in ModelTester.all_token_types:
                    fix_tokens_by_add = test_tokens.copy()
                    fix_tokens_by_add.append(add_token)
                    str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                    new_score_by_add = ModelTester.ngram.score(
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
                new_score_by_add = ModelTester.ngram.score(
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
                new_score_by_mod = ModelTester.ngram.score(
                    str_fix_by_mod_tokens)
                fix_prob.append(
                    (new_score_by_mod, fix_tokens_by_mod, "MOD", token_idx, mod_token))

            # try deleting token
            fix_tokens_by_del = test_tokens.copy()
            fix_tokens_by_del.pop(token_idx)
            str_fix_by_del_tokens = " ".join(fix_tokens_by_del)
            new_score_by_del = ModelTester.ngram.score(str_fix_by_del_tokens)
            fix_prob.append(
                (new_score_by_del, fix_tokens_by_del, "DEL", token_idx, to_change_token))

        fix_prob.sort(key=lambda x: x[0], reverse=True)
        return (source_path, fix_prob)

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
    def _hmm_locate_and_fix(source_and_err_tokens, model_name="n/a"):

        if model_name == "atn":
            SCORE_FUNC = ModelTester._atn_score
        elif model_name == "rule":
            SCORE_FUNC = ModelTester._rule_score

        # LOCATE THE MOST LIKELY ERROR LOCATIONS USING THE HMM
        (source_path, test_tokens) = source_and_err_tokens  # unpack the dict item
        test_seq_ids = list(
            map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], test_tokens))
        token_idx_prob = []
        _locate_pre_add_fix = dict()
        _locate_add_fix_score = dict()
        for seq_idx in range(1, len(test_seq_ids)):
            _locate_pre_add_fix[seq_idx] = test_seq_ids[:seq_idx]

        with Pool() as pool:
            for (score, add_fix_seq_idx) in pool.imap(SCORE_FUNC, _locate_pre_add_fix.items()):
                _locate_add_fix_score[add_fix_seq_idx] = score
                ModelTester.logger.info(
                    "{path}: {name} SCORING_LOCATION {token_idx} ({score})".format(
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
                "{path}: {name} CHECKING_LOCATION_SCORE {token_idx} ({score})".format(
                    name=model_name,
                    path=source_path,
                    token_idx=seq_idx,
                    score=contrib_score
                ))
            token_idx_prob.append((seq_idx, contrib_score))
        token_idx_prob.sort(key=lambda x: x[1])

        # RECCOMEND LIKELY FIXES USING THE HMM
        fix_prob = []

        # do all adds for zero in case break is in the first element...
        _zero_add_fix = dict()
        for add_token in ModelTester.all_token_type_ids:
            fix_tokens_by_add = test_seq_ids.copy()
            fix_tokens_by_add.insert(0, add_token)
            _zero_add_fix[add_token] = fix_tokens_by_add
        with Pool() as pool:
            for (score, add_token) in pool.imap(SCORE_FUNC, _zero_add_fix.items()):
                ModelTester.logger.debug(
                    "{path}: CHECK_ADD {token_idx} BEFORE SEQUENCE ({score}) ({name}-hmm)".format(
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
                "{path}: CHECKING_LOCATION {token_idx} ({score}) ({name}-hmm)".format(
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
            with Pool() as pool:
                for (score, add_token) in pool.imap(SCORE_FUNC, _add_fix.items()):
                    ModelTester.logger.debug(
                        "{path}: CHECKING_ADD {token} AT {pos} ({score}) ({name}-hmm)".format(
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
            with Pool() as pool:
                for (score, mod_token) in pool.imap(SCORE_FUNC, _sub_fix.items()):
                    ModelTester.logger.debug(
                        "{path}: CHECKING_MOD {token} AT {pos} ({score}) ({name}-hmm)".format(
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
            if model_name == "rule":
                new_score_by_del = ModelTester.rule_hmm.score(fix_tokens_by_del)
            elif model_name == "atn":
                new_score_by_del = ModelTester.atn_hmm.score(fix_tokens_by_del)
            ModelTester.logger.debug(
                "{path}: CHECKING_DEL {token} AT {pos} ({score}) ({name}-hmm)".format(
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
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="rule")

    @staticmethod
    def _atn_hmm_locate_and_fix(source_and_err_tokens):
        return ModelTester._hmm_locate_and_fix(source_and_err_tokens, model_name="atn")

    @staticmethod
    def _mean_reciprocal_rank(ranks):
        """
        Take all ranks, calculate meanreciprocal rank.
        Ranks are integers ranging from 1 to {num_suggestions}
        """
        reciprocal_ranks = map(lambda x: 1/x, ranks)
        return sum(reciprocal_ranks)/len(ranks)

    @classmethod
    def init_models(cls):
        cls.ngram = KenLM10Gram()
        cls.atn_hmm = ATNJavaTokenHMM()
        cls.rule_hmm = RuleJavaTokenHMM()

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

        # PERFORM NGRAM LOCATION DETECTION AND FIX RECCOMENDATION
        ngram_ranks = []
        with Pool() as pool:
            for (source_path, fix_probs) in pool.imap(ModelTester._ngram_locate_and_fix, self.one_error_tokens.items()):
                eval_stub = self.eval_str[source_path]
                orig_seq = self.file_to_tokens[source_path]
                rank = 1
                for fix_prob in fix_probs:
                    # unpack
                    new_score, fix_sequence, action, token_idx, to_change_token = fix_prob
                    ModelTester.logger.info("NGRAM: RANK %d SUGGEST %s %s AT %d (score: %.2f) %s", rank, action,
                                            to_change_token, token_idx, new_score, eval_stub)
                    if fix_sequence == orig_seq:
                        break
                    else:
                        rank += 1
                ngram_ranks.append(rank)

        # PERFORM RULE HMM LOCATION DETECTION AND FIX RECCOMENDATION
        rule_ranks = []
        for (source_path, fix_probs) in map(ModelTester._rule_hmm_locate_and_fix, self.one_error_tokens.items()):
            eval_stub = self.eval_str[source_path]
            orig_seq = self.file_to_tokens[source_path]
            orig_seq_ids = list(
                map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], orig_seq))
            rank = 1
            for fix_prob in fix_probs:
                # unpack
                new_score, fix_sequence, action, token_idx, to_change_token = fix_prob
                ModelTester.logger.info("RULE: RANK %d, SUGGEST %s %s AT %d (score: %.2f) %s", rank, action,
                                        SourceCodeParser.JAVA_TOKEN_ID_MAP[to_change_token], token_idx, new_score, eval_stub)
                if fix_sequence == orig_seq_ids:
                    break
                else:
                    rank += 1
            rule_ranks.append(rank)

        # PERFORM ATN HMM LOCATION DETECTION AND FIX RECCOMENDATION
        atn_ranks = []
        for (source_path, fix_probs) in map(ModelTester._atn_hmm_locate_and_fix, self.one_error_tokens.items()):
            eval_stub = self.eval_str[source_path]
            orig_seq = self.file_to_tokens[source_path]
            orig_seq_ids = list(
                map(lambda x: SourceCodeParser.JAVA_TOKEN_TYPE_MAP[x], orig_seq))
            rank = 1
            for fix_prob in fix_probs:
                # unpack
                new_score, fix_sequence, action, token_idx, to_change_token = fix_prob
                ModelTester.logger.info("ATN: RANK %d, SUGGEST %s %s AT %d (score: %.2f) %s", rank, action,
                                        SourceCodeParser.JAVA_TOKEN_ID_MAP[to_change_token], token_idx, new_score, eval_stub)
                if fix_sequence == orig_seq_ids:
                    break
                else:
                    rank += 1
            atn_ranks.append(rank)

        ngram_mrr = ModelTester._mean_reciprocal_rank(ngram_ranks)
        ngram_found = [x for x in ngram_ranks if x <
                       ModelTester.num_suggestions]
        print("ngram found {found}/{total} true fixes (Mean Reciprocal Rank={mrr:.2})".format(
            found=len(ngram_found), total=len(ngram_ranks), mrr=ngram_mrr))

        rule_mrr = ModelTester._mean_reciprocal_rank(rule_ranks)
        rule_found = [x for x in rule_ranks if x < ModelTester.num_suggestions]
        print("rule-hmm found {found}/{total} true fixes (Mean Reciprocal Rank={mrr:.2})".format(
            found=len(rule_found), total=len(rule_ranks), mrr=rule_mrr))

        atn_mrr = ModelTester._mean_reciprocal_rank(atn_ranks)
        atn_found = [x for x in atn_ranks if x < ModelTester.num_suggestions]
        print("atn-hmm found {found}/{total} true fixes (Mean Reciprocal Rank={mrr:.2})".format(
            found=len(atn_found), total=len(atn_ranks), mrr=atn_mrr))
