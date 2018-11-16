#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""

import sqlite3
from argparse import ArgumentParser
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker

from grammar.JavaParser import JavaParser
from grammar.JavaLexer import JavaLexer
from grammar.JavaParserListener import JavaParserListener

class KeyPrinter(JavaParserListener):
    """
    Tree stepper class for printing rules, hacked on from
    https://www.antlr.org/api/Java/org/antlr/v4/runtime/tree/ParseTreeListener.html
    """
    OPEN_RULE = "{"
    CLOSE_RULE = "}"
    SPACER = "  "

    def __init__(self, verbose, **kwargs):
        self.verbose = verbose
        self.literal_rule_stack = []
        self.symbolic_rule_stack = []
        super().__init__(**kwargs)

    def enterEveryRule(self, ctx):
        """override"""
        self.symbolic_rule_stack.append(
            self.OPEN_RULE + "<R:" + str(ctx.getRuleIndex()) + ">"
        )
        self.literal_rule_stack.append(
            self.OPEN_RULE + "<R:" + type(ctx).__name__ + ">"
        )

    def exitEveryRule(self, ctx):
        """override"""
        self.symbolic_rule_stack.append(self.CLOSE_RULE)
        self.literal_rule_stack.append(self.CLOSE_RULE)

    def visitTerminal(self, node):
        """override"""
        self.symbolic_rule_stack.append("<T:" + str(node.symbol.type) + ">")
        self.literal_rule_stack.append("<T:" + node.symbol.text + ">")

    def show_sequence(self):
        """Print out the rule stack"""
        if self.verbose:
            depth = 0
            for text in self.literal_rule_stack:
                if self.OPEN_RULE in text:
                    print((self.SPACER * depth) + text)
                    depth += 1
                elif self.CLOSE_RULE in text:
                    depth -= 1
                    print((self.SPACER * depth) + text)
                else:
                    print((self.SPACER * depth) + text)
        else:
            print(" ".join(self.symbolic_rule_stack))

def dummy_parse_example(offset=0, verbose=False):
    """
    Quick example of taking Java source code and outputting rules/tokens
    """
    conn = sqlite3.connect("data/java-sources-20170628.sqlite3")
    cursor = conn.execute("SELECT * FROM source_file LIMIT 1 OFFSET (?)", (offset,))
    row = cursor.fetchone()
    conn.close()

    source_code = row[1].decode("utf-8")
    source_buf = InputStream(source_code)
    lexer = JavaLexer(source_buf)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()

    print("============== SOURCE CODE ==============")
    print(source_code)
    print()

    print("============== PARSED CODE ==============")
    printer = KeyPrinter(verbose)
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    printer.show_sequence()

def main():
    """INSYN script function. Currently only parses sources into grammar trees."""
    parser = ArgumentParser(
        description="Reccomendation Models for Syntactically Incorrect Source Code"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "-p", "--parse-example",
        help="example output sequence from sqlite3 db",
        action="store_true"
    )
    parser.add_argument(
        "-o", "--db-offset",
        help="in example, number of sqlite3 db rows to offset",
        action="store",
        default=0
    )
    args = parser.parse_args()

    if args.parse_example:
        dummy_parse_example(args.db_offset, args.verbose)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
