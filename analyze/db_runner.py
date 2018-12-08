"""Classes and methods for processing rows in the database
"""
import logging
import os.path
import tables
import sqlite3
import sys
import time
import numpy as np
from multiprocessing import Pool
from antlr4 import ParseTreeWalker

from analyze.parser import SourceCodeParser
from grammar.JavaParser import JavaParser
from grammar.tree_helper import ParseTreeStepper


class DBRunner:
    logger = logging.getLogger(__name__)
    db_path = os.path.join(os.path.dirname(__file__),
                           "..", "data", "java-sources-20170628.sqlite3")

    def __init__(self, verbose=True):
        self.verbose = verbose
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError("Missing {0}".format(self.db_path))

    @staticmethod
    def _javac_init():
        """
        Called by pool initializer in tokenize_all_db_source.
        Setup SourceCodeParser javac instance (py4j JavaGateway binding)
        """
        global sc_parser
        sc_parser = SourceCodeParser()

    @staticmethod
    def _tokenize_sql_row_result(row_result):
        """
        Called within pool imap for tokenizing sqlite database source code.
        Returns a string of token ids seperated by spaces or logs an error to stderr.
        """
        (file_hash, raw_source_code) = row_result
        try:
            source_code = raw_source_code.decode("utf-8")
            (_, tokens) = sc_parser.javac_analyze(source_code)
            int_tokens = list(sc_parser.tokens_to_ints(tokens))
            if (-1) in int_tokens:
                joined_token_error = ", ".join(
                    map(lambda s: str(s), tokens[int_tokens.index(-1)]))
                DBRunner.logger.error(
                    "{filehash} contains token error: {token_error}".format(
                        filehash=file_hash, token_error=joined_token_error))
            else:
                # return " ".join(map(lambda itos: str(itos), int_tokens))
                return " ".join(map(lambda tup: str(tup[0]), tokens))
        except Exception as e:
            DBRunner.logger.error("{filehash} threw {err}".format(
                filehash=file_hash, err=str(e)))


    @staticmethod
    def _tokenize_with_rule_sql_row_result(row_result):
        """
        Get the Javac token AND corresponding rule at each token
        """
        (file_hash, raw_source_code) = row_result
        try:
            source_code = raw_source_code.decode("utf-8")
            (_, javac_tokens) = sc_parser.javac_analyze(source_code)
            int_tokens = list(sc_parser.tokens_to_ints(javac_tokens))
            if -1 in int_tokens:
                joined_token_error = ", ".join(
                    map(lambda s: str(s), javac_tokens[int_tokens.index(-1)]))
                DBRunner.logger.error(
                    "{filehash} contains token error: {token_error}".format(
                        filehash=file_hash, token_error=joined_token_error))
                return
            javac_token_to_literal = map(lambda tup: (tup[0], tup[1]), javac_tokens)
            (antlr_num_errors, _, tree) = SourceCodeParser.antlr_analyze(source_code)
            if antlr_num_errors > 0:
                DBRunner.logger.error(
                    "{filehash} contains {num_err} antlr errors, ignoring".format(
                        filehash=file_hash, num_err=antlr_num_errors))
                return
            printer = ParseTreeStepper(False)
            walker = ParseTreeWalker()
            walker.walk(printer, tree)
            return printer.unclean_map_javac_to_antlr(javac_token_to_literal)

        except Exception as e:
            DBRunner.logger.error("{filehash} threw {err}".format(
                filehash=file_hash, err=str(e)))

    def tokenize_all_db_source(self, output_type="name"):
        for token_seq in self.tokenize_all_db_source_gen(output_type=output_type):
            if output_type == "id":
                def map_func(name): return str(
                    SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(name, -1))
                map_out = map(map_func, token_seq.split())
                print(" ".join(map_out))
            else:
                print(token_seq)

    def tokenize_all_db_source_gen(self, output_type="name", limit=9999999):
        """
        Tokenize all source files, output to stdout (for training ngram)
        Uses stderr for printing progress messages and errors.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(hash) FROM source_file")
        num_rows = cursor.fetchone()[0]
        cursor.execute("SELECT hash, source FROM source_file LIMIT(?)",
                       (min(limit, num_rows),))
        row_results = cursor.fetchmany()
        counter = 0
        start_time = time.time()
        all_sql_results = []
        while row_results:
            for row_result in row_results:
                all_sql_results.append(row_result)
                if not counter % 2500:
                    elapsed_time = time.time() - start_time
                    DBRunner.logger.error(
                        "\rSQL READ: {counter:0{c_width}d}/{total} ({progress:.2%}) {seconds:.1f}s elapsed\r".format(
                            counter=counter,
                            c_width=len(str(num_rows)),
                            total=num_rows,
                            progress=counter/num_rows,
                            seconds=(elapsed_time)
                        )
                    )
                counter += 1
            row_results = cursor.fetchmany()

            # tokenization every number of rows to reduce memory usage
            tokenize_num_rows = len(all_sql_results)
            if tokenize_num_rows > 10000 or not row_results:
                tokenize_counter = 0
                with Pool(initializer=DBRunner._javac_init) as pool:
                    for str_tokens in pool.imap(DBRunner._tokenize_sql_row_result, all_sql_results):
                        if not tokenize_counter % 250:
                            elapsed_time = time.time() - start_time
                            DBRunner.logger.error(
                                "\r    TOKENIZE: {counter:0{c_width}d}/{total} of batch {sql_counter}/{sql_total_rows}; {seconds:.1f}s elapsed\r".format(
                                    counter=tokenize_counter,
                                    c_width=len(str(tokenize_num_rows)),
                                    total=tokenize_num_rows,
                                    progress=tokenize_counter/tokenize_num_rows,
                                    seconds=(elapsed_time),
                                    sql_counter=counter,
                                    sql_total_rows=num_rows
                                )
                            )
                        tokenize_counter += 1
                        if str_tokens:
                            if output_type == "id":
                                # map name to id
                                def map_func(name): return str(
                                    SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(name, -1))
                                map_out = map(map_func, str_tokens.split())
                                yield list(map_out)
                            elif output_type == "np_id":
                                # map name to id
                                def map_func(name): return str(
                                    SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(name, -1))
                                map_out = map(map_func, str_tokens.split())
                                yield np.array(list(map_out))
                            else:
                                # default
                                yield str_tokens

                all_sql_results = []
        conn.close()

    def tokenize_with_rule_db_sources(self):

        max_token_size = max(
            list(map(lambda x: len(x), SourceCodeParser.JAVA_TOKEN_TYPE_MAP.keys())))

        token_seq_filename = "token_sequences.h5"
        token_seq_file = tables.open_file(token_seq_filename, mode="w")
        token_file_array = token_seq_file.create_vlarray(
            token_seq_file.root, "data", atom=tables.StringAtom(max_token_size)
        )

        max_rule_size = max(map(lambda x: len(x), JavaParser.ruleNames))

        rule_seq_filename = "rule_sequences.h5"
        rule_seq_file = tables.open_file(rule_seq_filename, mode="w")
        rule_file_array = rule_seq_file.create_vlarray(
            rule_seq_file.root, "data", atom=tables.StringAtom(max_rule_size)
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(hash) FROM source_file")
        num_rows = cursor.fetchone()[0]
        cursor.execute("SELECT hash, source FROM source_file LIMIT(10)")
        row_results = cursor.fetchmany()
        counter = 0
        start_time = time.time()
        all_sql_results = []
        while row_results:
            for row_result in row_results:
                all_sql_results.append(row_result)
                if not counter % 2500:
                    elapsed_time = time.time() - start_time
                    DBRunner.logger.error(
                        "\rSQL READ: {counter:0{c_width}d}/{total} ({progress:.2%}) {seconds:.1f}s elapsed\r".format(
                            counter=counter,
                            c_width=len(str(num_rows)),
                            total=num_rows,
                            progress=counter/num_rows,
                            seconds=(elapsed_time)
                        )
                    )
                counter += 1
            row_results = cursor.fetchmany()

            # tokenization every number of rows to reduce memory usage
            tokenize_num_rows = len(all_sql_results)
            if tokenize_num_rows > 10000 or not row_results:
                tokenize_counter = 0
                with Pool(initializer=DBRunner._javac_init) as pool:
                    for token_types_to_rules in pool.imap(DBRunner._tokenize_with_rule_sql_row_result, all_sql_results):
                        if not tokenize_counter % 250:
                            elapsed_time = time.time() - start_time
                            DBRunner.logger.error(
                                "\r    TOKENIZE: {counter:0{c_width}d}/{total} of batch {sql_counter}/{sql_total_rows}; {seconds:.1f}s elapsed\r".format(
                                    counter=tokenize_counter,
                                    c_width=len(str(tokenize_num_rows)),
                                    total=tokenize_num_rows,
                                    progress=tokenize_counter/tokenize_num_rows,
                                    seconds=(elapsed_time),
                                    sql_counter=counter,
                                    sql_total_rows=num_rows
                                )
                            )
                        tokenize_counter += 1
                        if token_types_to_rules:
                            token_types = list(map(lambda x: x[0], token_types_to_rules))
                            rule_names = list(map(lambda x: x[1], token_types_to_rules))
                            token_file_array.append(np.array(token_types))
                            rule_file_array.append(np.array(rule_names))
                all_sql_results = []
        print("wrote to " + token_seq_filename)
        print("wrote to " + rule_seq_filename)
        token_seq_file.close()
        rule_seq_file.close()
        conn.close()

    def create_npy_train_data(self, size=10):
        # MAX_LEN = 2420215 # known because we tokenized all data once
        filename = "train_data_size_{}.h5".format(size)
        h5file = tables.open_file(filename, mode="w")
        array_c = h5file.create_vlarray(
            h5file.root, 'data', atom=tables.Int16Atom())
        # no_pad = []

        for np_arr in self.tokenize_all_db_source_gen(output_type="np_id", limit=size):
            # cur = np.full((1, MAX_LEN), np.nan)
            # cur[0][:len(np_arr)] = np_arr
            # array_c.append(cur)
            # no_pad.append(np_arr)

            array_c.append(np_arr)
        # np_arrs = np.array(no_pad, dtype=object)
        h5file.close()
        print("Saved to {}".format(filename))

    def view_one_db_source(self, offset=0):
        """
        Quick example of taking Java source code and outputting rules/tokens
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT source FROM source_file LIMIT 1 OFFSET (?)", (offset,))
        source_code = cursor.fetchone()[0].decode("utf-8")
        conn.close()

        if self.verbose:
            print("============== SOURCE CODE ==============")
        print(source_code)
        print()

        # analyze db code
        sc_parser = SourceCodeParser()
        (javac_num_err, javac_tokens,) = sc_parser.javac_analyze(source_code)
        (antlr_num_err, antlr_tokens, tree) = sc_parser.antlr_analyze(source_code)

        if self.verbose:
            print("============== PARSED CODE ==============")
            printer = ParseTreeStepper(self.verbose)
            walker = ParseTreeWalker()
            walker.walk(printer, tree)
            printer.show_sequence()

            print("============== META/TOKENS ==============")
            print("length of source code string: {0}".format(len(source_code)))
            print()

            print("---- ANTLR TOKENS ----")
            str_antlr_tokens = map(lambda antlr_token: (
                tree.parser.symbolicNames[antlr_token.type],
                # antlr_token.text
            ), antlr_tokens)
            print(list(str_antlr_tokens))
            print()

            print("---- JAVAC TOKENS ----")
            str_javac_tokens = map(lambda javac_token: (
                javac_token[0],
                # javac_token[1]
            ), javac_tokens)
            print(list(str_javac_tokens))
            print()

            print("============== NUM ERRORS ==============")
        print("antlr found {0} errors in {1} tokens".format(
            antlr_num_err, len(antlr_tokens)))
        print("javac found {0} errors in {1} tokens".format(
            javac_num_err, len(javac_tokens)))
