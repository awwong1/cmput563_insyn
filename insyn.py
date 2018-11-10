#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""

import sys
import sqlite3
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker

from grammar.JavaParser import JavaParser
from grammar.JavaLexer import JavaLexer
from grammar.JavaParserListener import JavaParserListener

class KeyPrinter(JavaParserListener):
    """
    Tree stepper class for printing rules, hacked on from
    https://www.antlr.org/api/Java/org/antlr/v4/runtime/tree/ParseTreeListener.html
    """
    def enterEveryRule(self, ctx):
        print("ENTER", type(ctx.getRuleContext()))
        #print("enterRule, invokingState", ctx.invokingState, ctx)
        #print("\tSTART:", ctx.start)
        #print("\tSTOP:", ctx.stop)

    def exitEveryRule(self, ctx):
        print("EXIT", type(ctx.getRuleContext()))
        #print("exitRule, invokingState", ctx.invokingState)
        #print("\tSTART:", ctx.start)
        #print("\tSTOP:", ctx.stop)
        pass

    def visitTerminal(self, node):
        print("\t", node)

def main(*args, **kwargs):
    """INSYN script function. Currently only parses sources into grammar trees."""

    conn = sqlite3.connect("data/java-sources-20170628.sqlite3")
    cursor = conn.execute("SELECT * FROM source_file LIMIT 1")
    row = cursor.fetchone()

    # print(cursor.description)
    # print(row)

    source_code = row[1].decode("utf-8")
    source_buf = InputStream(source_code)
    lexer = JavaLexer(source_buf)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()

    print(source_code)
    print()

    printer = KeyPrinter()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)

    conn.close()

if __name__ == "__main__":
    main()
