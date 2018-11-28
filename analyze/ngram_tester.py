import logging
import fnmatch
import os
import random

from analyze.parser import SourceCodeParser
from model.ngram import NGram


class NGramTester:
    """
    Take java source code, perform one token modification (add, remove, change),
    then use the ngram to suggest fix.
    """
    logger = logging.getLogger(__name__)
    pattern = "*.java"
    all_token_types = list(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())
    try_num_locations = 5  # How many locations do we try?

    def __init__(self, input_file_path):
        self.test_file_sources = []
        self.file_to_tokens = dict()  # original valid token sequence
        self.file_to_tests = dict()  # one change error sequences

        # get list of all source files to test
        if os.path.isfile(input_file_path):
            self.test_file_sources = fnmatch.filter(
                [input_file_path], self.pattern)
        elif os.path.isdir(input_file_path):
            for dirpath, _dirnames, filenames in os.walk(input_file_path):
                if not filenames:
                    continue
                match_files = fnmatch.filter(filenames, self.pattern)
                for match_file in match_files:
                    self.test_file_sources.append(
                        os.path.join(dirpath, match_file))

        # ensure all test source files are initially valid
        sc_parser = SourceCodeParser()
        for source_path in self.test_file_sources:
            with open(source_path, "r") as source_file:
                source_code = source_file.read()
                (num_errors, tuple_tokens) = sc_parser.javac_analyze(source_code)
                if num_errors:
                    NGramTester.logger.error("{num_errs} errors: skipping {path}".format(
                        num_errs=num_errors, path=source_path))
                else:
                    tokens_type = list(map(lambda tup: tup[0], tuple_tokens))
                    self.file_to_tokens[source_path] = tokens_type
                    self.file_to_tests[source_path] = tokens_type.copy()

        if not len(self.test_file_sources) or not len(self.file_to_tokens.keys()):
            raise FileNotFoundError("No valid {pattern} files found for {path}".format(
                pattern=self.pattern, path=input_file_path))

        self.ngram = NGram()

        # perform one random token change for all files
        for source_path, original_tokens in self.file_to_tokens.items():
            change_type = random.choice(["ADD", "DEL", "MOD"])
            change_idx = random.randrange(0, len(original_tokens))
            print(" ".join(original_tokens))
            if change_type == "ADD":
                rand_token = random.choice(self.all_token_types)
                self.file_to_tests[source_path].insert(
                    change_idx, rand_token)
                print("{}: {} {} at {}".format(
                    source_path, change_type, rand_token, change_idx))
            elif change_type == "DEL":
                self.file_to_tests[source_path].pop(change_idx)
                print("{}: {} {} at {}".format(
                    source_path, change_type, original_tokens[change_idx], change_idx))
            elif change_type == "MOD":
                sub_token_types = self.all_token_types.copy()
                sub_token_types.remove(original_tokens[change_idx])
                rand_token = random.choice(sub_token_types)
                self.file_to_tests[source_path][change_idx] = rand_token
                print("{}: {} from {} to {} at {}".format(
                    source_path, change_type, original_tokens[change_idx], rand_token, change_idx
                ))

        # use 10-gram to find likely location of error
        for source_path, test_tokens in self.file_to_tests.items():
            str_test_tokens = " ".join(test_tokens)
            counter = 0  # counter will go to full length of test_tokens due to </s>
            accum_score = 0

            token_idx_prob = []

            for prob, ngram_len, _ in self.ngram.full_scores(str_test_tokens):
                accum_score += prob
                NGramTester.logger.debug(
                    "{path}: {score} ({counter} of {total}) ngram={ngram_len}".format(
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
            for token_idx, prob in token_idx_prob[:self.try_num_locations]:
                to_change_token = test_tokens[token_idx]

                # try adding token
                for add_token in self.all_token_types:
                    fix_tokens_by_add = test_tokens.copy()
                    fix_tokens_by_add.insert(token_idx, add_token)
                    str_fix_by_add_tokens = " ".join(fix_tokens_by_add)
                    new_score_by_add = self.ngram.score(str_fix_by_add_tokens)
                    fix_prob.append(
                        (new_score_by_add, "ADD", token_idx, "{} before {}".format(add_token, to_change_token)))

                # try changing token
                for mod_token in self.all_token_types:
                    fix_tokens_by_mod = test_tokens.copy()
                    fix_tokens_by_mod[token_idx] = mod_token
                    str_fix_by_mod_tokens = " ".join(fix_tokens_by_mod)
                    new_score_by_mod = self.ngram.score(str_fix_by_mod_tokens)
                    fix_prob.append(
                        (new_score_by_mod, "MOD", token_idx, "{} into {}".format(to_change_token, mod_token)))

                # try deleting token
                fix_tokens_by_del = test_tokens.copy()
                fix_tokens_by_del.pop(token_idx)
                str_fix_by_del_tokens = " ".join(fix_tokens_by_del)
                new_score_by_del = self.ngram.score(str_fix_by_del_tokens)
                fix_prob.append(
                    (new_score_by_del, "DEL", token_idx, to_change_token))

            fix_prob.sort(key=lambda x: x[0], reverse=True)
            for (score, action, idx, token_data) in fix_prob:
                NGramTester.logger.info(
                    "suggest {action} {token_data} at {idx} (score: {score})".format(
                        action=action,
                        token_data=token_data,
                        idx=idx,
                        score=score
                    ))
