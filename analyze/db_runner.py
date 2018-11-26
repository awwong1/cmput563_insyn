"""Classes and methods for processing rows in the database
"""
import logging
import os.path
import sqlite3
import sys
import time

from analyze.parser import SourceCodeParser


class DatabaseRunner:
    logger = logging.getLogger(__name__)
    db_path = os.path.join("data", "java-sources-20170628.sqlite3")

    def __init__(self, verbose=True):
        self.verbose = verbose
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError("Missing {0}".format(self.db_path))

    def tokenize_all_db_source(self):
        """
        Tokenize all source files, output to standard out (for training ngram)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source FROM source_file")
        results = cursor.fetchmany()
        while results:
            for (raw_source_code,) in results:
                source_code = raw_source_code.decode("utf-8")
                (_, tokens) = SourceCodeParser.javac_analyze(source_code)
                int_tokens = SourceCodeParser.tokens_to_ints(tokens)
                print(" ".join(map(lambda itos: str(itos), int_tokens)))
            results = cursor.fetchmany()
        conn.close()

    def view_all_db_source(self):
        """
        Validate dataset parses correctly.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(hash) FROM source_file")
        num_rows = cursor.fetchone()[0]
        # get all source file training data
        cursor.execute("SELECT hash, source FROM source_file")
        results = cursor.fetchmany()

        counter = 0
        start_time = time.time()
        while results:  # and counter < 1000:
            for (file_hash, raw_source_code) in results:
                source_code = raw_source_code.decode("utf-8")
                (javac_num_errs, javac_tokens) = SourceCodeParser.javac_analyze(
                    source_code)
                if javac_num_errs:
                    self.logger.warning(
                        "{0} javac found {1} errors in {2} tokens".format(
                            file_hash, javac_num_errs, len(javac_tokens))
                    )
                elapsed_time = time.time() - start_time
                if self.verbose:
                    self.logger.info(
                        "{counter:0{c_width}d}/{total} ({progress:.2%}) {seconds:.1f}s elapsed\r".format(
                            counter=counter,
                            c_width=len(str(num_rows)),
                            total=num_rows,
                            progress=counter/num_rows,
                            seconds=(elapsed_time)
                        )
                    )
                counter += 1
            results = cursor.fetchmany()
        print("\nFinished!".format())
        conn.close()

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
        (javac_num_err, javac_tokens,) = SourceCodeParser.javac_analyze(source_code)
        (antlr_num_err, antlr_tokens,
         tree) = SourceCodeParser.antlr_analyze(source_code)

        if self.verbose:
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
