#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""
import logging
from argparse import ArgumentParser, SUPPRESS

from analyze.db_runner import DatabaseRunner
from analyze.parser import SourceCodeParser
from grammar.structure import StructureGenerator
from model.ngram import NGram

def main():
    """INSYN script function. Currently only parses sources into grammar trees."""
    parser = ArgumentParser(
        description="Reccomendation Models for Syntactically Incorrect Source Code"
    )
    parser.add_argument(
        "-l", "--log",
        help="set logging verbosity",
        metavar="level",
        default="warning"
    )
    parser.add_argument(
        "-f", "--fix",
        help="if applicable list all possible one token fixes",
        metavar="input.java",
        nargs=1,
        action="store"
    )
    parser.add_argument(
        "--sample-parse",
        help="sample output sequence from training db",
        default=SUPPRESS,
        metavar="offset",
        nargs='?',
        type=int,
        action="store"
    )
    parser.add_argument(
        "--generate-structure",
        help="generate HHMM structure from grammar",
        action="store_true"
    )
    parser.add_argument(
        "--tokenize-training-data",
        help="tokenize all training data",
        action="store_true"
    )
    parser.add_argument(
        "--prob-test",
        help="work in progress tool > probabilities",
        action="store_true"
    )

    args = parser.parse_args()
    if args.log:
        raw_log_level = str(args.log).upper()
        if hasattr(logging, raw_log_level):
            log_level = getattr(logging, raw_log_level)
            logging.basicConfig(level=log_level)
        else:
            logging.warning("Invalid log level: {0}".format(args.log))
    else:
        logging.basicConfig()

    if hasattr(args, "sample_parse"):
        DatabaseRunner().view_one_db_source(args.sample_parse or 0)
    elif args.generate_structure:
        StructureGenerator()
    elif args.fix:
        # SourceCodeParser(args.fix[0])
        print("todo")
    elif args.tokenize_training_data:
        DatabaseRunner().tokenize_all_db_source()
    elif args.prob_test:
        print("todo")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
