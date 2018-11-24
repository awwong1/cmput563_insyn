import time
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from javac_parser import Java

from grammar.JavaParser import JavaParser
from grammar.JavaLexer import JavaLexer
from grammar.tree_helper import CountedDefaultErrorStrategy, ParseTreeStepper

class SourceCodeParser:
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

        tree = parser.compilationUnit()
        antlr_tokens = stream.tokens
        num_errors = parser._errHandler.num_errors

        return (num_errors, antlr_tokens, tree)

    @staticmethod
    def javac_analyze(source_code):
        num_errors = SourceCodeParser.javac.get_num_parse_errors(source_code)
        token_sequence = SourceCodeParser.javac.lex(source_code)
        # print(SourceCodeParser.javac.check_syntax(source_code))

        return (num_errors, token_sequence)

