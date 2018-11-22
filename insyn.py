#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""

import random
import sys
from argparse import ArgumentParser, SUPPRESS

from analyze.db_handler import view_one_db_source
from analyze.fixer import FixFinder
from grammar.structure import StructureGenerator


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
        "-p", "--parse-db",
        help="example output sequence from sqlite3 db",
        default=SUPPRESS,
        metavar="offset",
        nargs='?',
        action="store"
    )
    parser.add_argument(
        "-g", "--generate-structure",
        help="generate HHMM structure from grammar",
        action="store_true"
    ),
    parser.add_argument(
        "-f", "--fix",
        help="if applicable list all possible one token fixes",
        metavar="input.java",
        nargs=1,
        action="store"
    )
    args = parser.parse_args()

    if hasattr(args, "parse_db"):
        view_one_db_source(args.parse_db or 0, verbose=args.verbose)
    elif args.generate_structure:
        StructureGenerator(args.verbose)
    elif args.fix:
        FixFinder(args.fix[0])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
