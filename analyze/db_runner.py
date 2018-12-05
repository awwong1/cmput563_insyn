"""Classes and methods for processing rows in the database
"""
import logging
import os.path
import sqlite3
import sys
import time
import numpy as np 
from multiprocessing import Pool
from antlr4 import ParseTreeWalker

from analyze.parser import SourceCodeParser
from model.ngram import KenLM10Gram
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
                joined_token_error = ", ".join(map(lambda s: str(s), tokens[int_tokens.index(-1)]))
                DBRunner.logger.error(
                    "{filehash} contains token error: {token_error}".format(
                        filehash=file_hash, token_error=joined_token_error))
            else:
                #return " ".join(map(lambda itos: str(itos), int_tokens))
                return " ".join(map(lambda tup: str(tup[0]), tokens))
        except Exception as e:
            DBRunner.logger.error("{filehash} threw {err}".format(
                filehash=file_hash, err=str(e)))

    def tokenize_all_db_source(self, output_type="name"):
        """
        Tokenize all source files, output to stdout (for training ngram)
        Uses stderr for printing progress messages and errors.
        """

        raw_ids = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(hash) FROM source_file")
        num_rows = cursor.fetchone()[0]
        cursor.execute("SELECT hash, source FROM source_file")
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
                                map_func = lambda name: str(SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(name, -1))
                                map_out = map(map_func, str_tokens.split())
                                raw_ids.append(np.array(list(map_out))) 
                                # print(" ".join(map_out)) 
                            else:
                                # default
                                print(str_tokens)

                all_sql_results = []
        conn.close()
        np_ids = np.array(raw_ids, dtype=object) 
        np.save("train_data.npy", np_ids) 

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
                tree.parser.symbolicNames[antlr_token.type], antlr_token.text), antlr_tokens)
            print(list(str_antlr_tokens))
            print()

            print("---- JAVAC TOKENS ----")
            str_javac_tokens = map(lambda javac_token: (
                javac_token[0], javac_token[1]), javac_tokens)
            print(list(str_javac_tokens))
            print()

            print("============== NUM ERRORS ==============")
        print("antlr found {0} errors in {1} tokens".format(
            antlr_num_err, len(antlr_tokens)))
        print("javac found {0} errors in {1} tokens".format(
            javac_num_err, len(javac_tokens)))
