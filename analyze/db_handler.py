"""Classes and methods for processing rows in the database
"""
import random
import time
import sqlite3
from javac_parser import Java
from antlr4 import ParseTreeWalker

from analyze.fixer import FixFinder
from grammar.tree_helper import ParseTreeStepper



def view_one_db_source(offset=0, verbose=False):
    """
    Quick example of taking Java source code and outputting rules/tokens
    """
    conn = sqlite3.connect("data/java-sources-20170628.sqlite3")
    cursor = conn.execute("SELECT * FROM source_file LIMIT 1 OFFSET (?)", (offset,))
    row = cursor.fetchone()
    conn.close()

    source_code = row[1].decode("utf-8")
    # len_text = len(source_code)
    # err_idx = random.randrange(len_text)
    # source_code = source_code[:err_idx] + "~💩!" + source_code[err_idx:]

    print("============== SOURCE CODE ==============")
    print(source_code)
    print()

    # analyze db code
    (javac_num_errs, javac_tokens, javac_sec) = FixFinder.javac_analyze(source_code)
    (antlr_num_errs, antlr_tokens, antlr_sec, tree) = FixFinder.antlr_analyze(source_code)

    tree_stepper = ParseTreeStepper(verbose)
    walker = ParseTreeWalker()
    walker.walk(tree_stepper, tree)

    # print("============== ANTLR TREE ==============")
    # tree_stepper = ParseTreeStepper(verbose)
    # walker = ParseTreeWalker()
    # walker.walk(tree_stepper, tree)
    # tree_stepper.show_sequence()
    # print()

    print("============== META/TOKENS ==============")
    print("length of source code string: {0}".format(len(source_code)))
    print()

    print("---- ANTLR TOKENS ----")
    str_antlr_tokens = map(lambda antlr_token: (tree.parser.symbolicNames[antlr_token.type], antlr_token.text), antlr_tokens)
    print(list(str_antlr_tokens))
    print()

    print("---- JAVAC TOKENS ----")
    str_javac_tokens = map(lambda javac_token: (javac_token[0], javac_token[1]), javac_tokens)
    print(list(str_javac_tokens))
    print()

    print("============== NUM ERRORS ==============")
    print("antlr took {0:.2g}s and found {1} errors for {2} tokens".format(antlr_sec, antlr_num_errs, len(antlr_tokens)))
    print("javac took {0:.2g}s and found {1} errors for {2} tokens".format(javac_sec, javac_num_errs, len(javac_tokens)))
