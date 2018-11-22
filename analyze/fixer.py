import time
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from javac_parser import Java

from grammar.JavaParser import JavaParser
from grammar.JavaLexer import JavaLexer
from grammar.tree_helper import CountedDefaultErrorStrategy, ParseTreeStepper

class FixFinder:

    javac = Java()

    def __init__(self, filename):
        try:
            self.source_file = open(filename, "r")
        except FileNotFoundError as err:
            print(err)

    @staticmethod
    def antlr_analyze(source_code):
        source_buf = InputStream(source_code)
        lexer = JavaLexer(source_buf)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(input=stream)
        parser._errHandler = CountedDefaultErrorStrategy()

        antlr_start = time.time()
        tree = parser.compilationUnit()
        antlr_end = time.time()
        elapsed_seconds = (antlr_end - antlr_start) / (10 ** 9)
        antlr_tokens = stream.tokens
        num_errors = parser._errHandler.num_errors

        return (num_errors, antlr_tokens, elapsed_seconds, tree)

    @staticmethod
    def javac_analyze(source_code):
        javac_start = time.time()
        num_errors = FixFinder.javac.get_num_parse_errors(source_code)
        javac_end = time.time()
        elapsed_seconds = (javac_end - javac_start) / (10 ** 9)
        token_sequence = FixFinder.javac.lex(source_code)
        # print(FixFinder.javac.check_syntax(source_code))

        return (num_errors, token_sequence, elapsed_seconds)

