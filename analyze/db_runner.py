"""Classes and methods for processing rows in the database
"""
import logging
import os.path
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count, current_process

from analyze.parser import SourceCodeParser

async_parsers = dict()  # holds the javac stuff per process maybe

class DatabaseRunner:
    logger = logging.getLogger(__name__)
    db_path = os.path.join(os.path.dirname(__file__),
                           "..", "data", "java-sources-20170628.sqlite3")

    def __init__(self, verbose=True):
        self.verbose = verbose
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError("Missing {0}".format(self.db_path))

    @staticmethod
    def _javac_init():
        # big old hack for getting stateful processes with javac py4j nonsense
        global sc_parser
        sc_parser = SourceCodeParser()

    @staticmethod
    def _tokenize_sql_row_result(row_result):
        # logging.error("proc: {name}".format(name=current_process().name))
        # needs the sc_parser global that was initialized
        (file_hash, raw_source_code) = row_result
        try:
            source_code = raw_source_code.decode("utf-8")
            (_, tokens) = sc_parser.javac_analyze(source_code)
            int_tokens = list(sc_parser.tokens_to_ints(tokens))
            if (-1) in int_tokens:
                logging.error(
                    "{filehash} contains token error".format(filehash=file_hash))
            else:
                return " ".join(map(lambda itos: str(itos), int_tokens))
        except Exception as e:
            logging.error("{filehash} threw {err}".format(
                filehash=file_hash, err=str(e)))

    def tokenize_all_db_source(self):
        """
        Tokenize all source files, output to standard out (for training ngram)
        """
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
                if not counter % 1000:
                    elapsed_time = time.time() - start_time
                    logging.error(
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
        conn.close()

        num_rows = len(all_sql_results)
        counter = 0
        with Pool(processes=cpu_count(), initializer=DatabaseRunner._javac_init) as pool:
            for str_tokens in pool.imap(DatabaseRunner._tokenize_sql_row_result, all_sql_results):
                if not counter % 100:
                    elapsed_time = time.time() - start_time
                    logging.error(
                        "\rTOKENIZE: {counter:0{c_width}d}/{total} ({progress:.2%}) {seconds:.1f}s elapsed\r".format(
                            counter=counter,
                            c_width=len(str(num_rows)),
                            total=num_rows,
                            progress=counter/num_rows,
                            seconds=(elapsed_time)
                        )
                    )
                counter += 1
                if str_tokens:
                    print(str_tokens)

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
        sc_parser = SourceCodeParser()
        while results:  # and counter < 1000:
            for (file_hash, raw_source_code) in results:
                source_code = raw_source_code.decode("utf-8")
                (javac_num_errs, javac_tokens) = sc_parser.javac_analyze(
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
        sc_parser = SourceCodeParser()
        (javac_num_err, javac_tokens,) = sc_parser.javac_analyze(source_code)
        (antlr_num_err, antlr_tokens, tree) = sc_parser.antlr_analyze(source_code)

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
