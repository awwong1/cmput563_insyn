import logging
import fnmatch
import os
import random
from multiprocessing import Pool, cpu_count, current_process
from analyze.parser import SourceCodeParser
from model.ngram import KenLM10Gram


class NGramTester:
    """
    Take java source code, perform one token modification (add, remove, change),
    then use the ngram to suggest fix.
    """
    logger = logging.getLogger(__name__)
    ngram = None # set in a class method call
    pattern = "*.java"
    all_token_types = list(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())
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
                NGramTester.logger.error("{path} contains {num_errs} error(s), skipping".format(
                    num_errs=num_errors, path=source_path),  # sc_parser.javac_check_syntax(source_code)
                )
            else:
                # return source_path, map(lambda x: x[0], tuple_tokens) # pickle fails with lambda
                return source_path, list(map(NGramTester._get_first, tuple_tokens))

    @staticmethod
    def _perform_one_token_break(source_path, original_tokens, test_tokens):
        """
        randomly perforn an ADD, DELETE or MODIFY on the original tokens sequence, put into test_tokens sequence
        """
        change_type = random.choice(["ADD", "DEL", "MOD"])
        change_idx = random.randrange(0, len(original_tokens))
        NGramTester.logger.debug(" ".join(original_tokens))
        token = "ERROR"  # set in the 3 change types
        if change_type == "ADD":
            rand_token = random.choice(NGramTester.all_token_types)
            test_tokens.insert(change_idx, rand_token)
            NGramTester.logger.info("{}: BREAK by {} {} at {}".format(source_path,
                                                              change_type, rand_token, change_idx))
            token = rand_token
        elif change_type == "DEL":
            token = test_tokens.pop(change_idx)
            NGramTester.logger.info("{}: BREAK by{} {} at {}".format(source_path, change_type,
                                                              original_tokens[change_idx], change_idx))
        elif change_type == "MOD":
            sub_token_types = NGramTester.all_token_types.copy()
            sub_token_types.remove(original_tokens[change_idx])
            rand_token = random.choice(sub_token_types)
            test_tokens[change_idx] = rand_token
            NGramTester.logger.info("{}: BREAK by {} from {} to {} at {}".format(
                source_path, change_type, original_tokens[change_idx], rand_token, change_idx
            ))
            token = rand_token
        return (change_type, change_idx, token)

    @staticmethod
    def _find_probabalistic_error_location(source_path, test_tokens):
        str_test_tokens = " ".join(test_tokens)
        counter = 0  # counter will go to full length of test_tokens due to </s>
        accum_score = 0

        token_idx_prob = []

        for prob, ngram_len, _ in NGramTester.ngram.full_scores(str_test_tokens):
            accum_score += prob
            NGramTester.logger.debug(
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
        return token_idx_prob

    @staticmethod
    def _suggest_probabalistic_fixes(source_path, test_tokens, token_idx_prob):
        fix_prob = []
        # For the most likely error locations, try add, mod, and delete
        for token_idx, score in token_idx_prob[:NGramTester.try_num_locations]:
            NGramTester.logger.info(
                "{path}: CHECKING_LOCATION {token_idx} ({score})".format(
                    path=source_path,
                    token_idx=token_idx,
                    score=score
                ))
            if token_idx == len(test_tokens):
                # edge case, we cannot modify or delete </s>, only add to end of sequence
                for add_token in NGramTester.all_token_types:
                    fix_tokens_by_add = test_tokens.copy()
                    fix_tokens_by_add.append(add_token)
                    str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                    new_score_by_add = NGramTester.ngram.score(
                        str_fix_by_add_tokens)
                    fix_prob.append(
                        (new_score_by_add, "ADD", token_idx, add_token))
                continue

            to_change_token = test_tokens[token_idx]
            # try adding token
            for add_token in NGramTester.all_token_types:
                fix_tokens_by_add = test_tokens.copy()
                fix_tokens_by_add.insert(token_idx, add_token)
                str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                new_score_by_add = NGramTester.ngram.score(
                    str_fix_by_add_tokens)
                fix_prob.append(
                    (new_score_by_add, "ADD", token_idx, add_token))

            # try changing token, cannot mod into itself
            sub_token_types = NGramTester.all_token_types.copy()
            sub_token_types.remove(to_change_token)
            for mod_token in sub_token_types:
                fix_tokens_by_mod = test_tokens.copy()
                fix_tokens_by_mod[token_idx] = mod_token
                str_fix_by_mod_tokens = " ".join(fix_tokens_by_mod)
                new_score_by_mod = NGramTester.ngram.score(
                    str_fix_by_mod_tokens)
                fix_prob.append(
                    (new_score_by_mod, "MOD", token_idx, mod_token))

            # try deleting token
            fix_tokens_by_del = test_tokens.copy()
            fix_tokens_by_del.pop(token_idx)
            str_fix_by_del_tokens = " ".join(fix_tokens_by_del)
            new_score_by_del = NGramTester.ngram.score(str_fix_by_del_tokens)
            fix_prob.append(
                (new_score_by_del, "DEL", token_idx, to_change_token))

        fix_prob.sort(key=lambda x: x[0], reverse=True)
        return fix_prob

    @staticmethod
    def _break_and_eval(source_and_tokens):
        """
        async safe break valid source code and evaluate model fix
        """
        (source_path, original_tokens) = source_and_tokens  # unpack the dict item
        test_tokens = original_tokens.copy()

        # break valid code by performing one random token change for all files
        (change_type, change_idx, change_token) = NGramTester._perform_one_token_break(
            source_path, original_tokens, test_tokens)

        # use 10-gram to find likely location of error
        token_idx_prob = NGramTester._find_probabalistic_error_location(
            source_path, test_tokens)

        # use 10-gram to suggest fixes for given locations
        fix_prob = NGramTester._suggest_probabalistic_fixes(
            source_path, test_tokens, token_idx_prob)

        eval_str = "{change} {c_token} at {t_idx}".format(
            change=change_type, c_token=change_token, t_idx=change_idx
        )

        # What was the break; Was true fix suggested and if so what rank?
        rank = 1
        for (score, action, idx, fix_token) in fix_prob[:NGramTester.num_suggestions]:
            NGramTester.logger.info(
                "{path}: SUGGEST_FIX_SCORE {score} ({action} {token} at {idx})".format(
                    path=source_path,
                    score=score,
                    action=action,
                    token=fix_token,
                    idx=idx
                ))
            eval_tokens = test_tokens.copy()
            if action == "ADD":
                if idx == len(eval_tokens):
                    eval_tokens.append(fix_token)  # </s> case
                else:
                    eval_tokens.insert(idx, fix_token)
            elif action == "MOD":
                eval_tokens[idx] = fix_token
            elif action == "DEL":
                eval_tokens.pop(idx)
            if eval_tokens == original_tokens:
                NGramTester.logger.info(
                    "{path}: TRUE_FIX_FOUND rank: {rank}".format(path=source_path, rank=rank))
                break
            rank += 1
        if rank <= NGramTester.num_suggestions:
            return source_path, eval_str + "; True fix found rank {}".format(rank), True, rank
        else:
            return source_path, eval_str + "; True fix not found, over rank {}".format(NGramTester.num_suggestions), False, rank

    @staticmethod
    def _mean_reciprocal_rank(ranks):
        """
        Take all ranks, calculate meanreciprocal rank.
        Ranks are integers ranging from 1 to {num_suggestions}
        """
        reciprocal_ranks = map(lambda x: 1/x, ranks)
        return sum(reciprocal_ranks)/len(ranks)

    @classmethod
    def init_ngram(cls):
        cls.ngram = KenLM10Gram()

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
        with Pool(initializer=NGramTester._javac_init) as pool:
            for path_and_tokens in pool.imap(NGramTester._tokenize_file_contents, self.file_paths):
                if path_and_tokens:
                    (source_path, tokens_type) = path_and_tokens
                    if tokens_type:
                        self.file_to_tokens[source_path] = list(tokens_type)

        if not len(self.file_to_tokens.keys()):
            raise ValueError("All {pattern} files syntactically incorrect".format(
                pattern=self.pattern))

    def run_evaluation(self):
        self._pre_evaluation()

        evaluation_counter = 0
        found_fix_counter = 0
        all_ranks = []  # if true fix found, what was the rank?
        with Pool() as pool:
            for (source_path, evaluation, found_fix, rank) in pool.imap(NGramTester._break_and_eval, self.file_to_tokens.items()):
                NGramTester.logger.info("{}: {}".format(source_path, evaluation))
                evaluation_counter += 1
                all_ranks.append(rank)
                if found_fix:
                    found_fix_counter += 1
        mrr = NGramTester._mean_reciprocal_rank(all_ranks)
        print("Found {found}/{total} true fixes (Mean Reciprocal Rank={mrr:.2})".format(found=found_fix_counter, total=evaluation_counter, mrr=mrr))
