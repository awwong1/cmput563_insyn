#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""
import logging
import numpy as np
from argparse import ArgumentParser, SUPPRESS

from analyze.db_runner import DBRunner
from analyze.parser import SourceCodeParser
from analyze.ngram_tester import NGramTester
from grammar.structure import StructureBuilder
from model.hmm import ATNJavaTokenHMM, RuleJavaTokenHMM, RuleJavaTokenHMMTrain


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
        "--test-ngram-model",
        help="read java code, change random token, list suggestions",
        metavar="file|dir",
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
        help="stdout training data (name='EOF', id='1')",
        nargs=1,
        choices=["name", "id"],
        action="store"
    )
    parser.add_argument(
        "--test-atn-hmm-model",
        help="read java code, change random token, list suggestions",
        metavar="file|dir",
        action="store"
    )
    parser.add_argument(
        "--test-rule-hmm-model",
        help="read java code, change random token, list suggestions",
        metavar="file|dir",
        action="store"
    )
    parser.add_argument(
        "--test-rule-hmm-model-train",
        help="read java code, change random token, list suggestions",
        metavar="file|dir",
        action="store"
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
        DBRunner().view_one_db_source(args.sample_parse or 0)
    elif args.test_ngram_model:
        NGramTester.init_ngram()
        NGramTester(args.test_ngram_model).run_evaluation()
    elif args.generate_structure:
        struct_builder = StructureBuilder()
        print("Building ATN transition and emissions...")
        atn_trans, atn_em = struct_builder.build_atn_hmm_matrices()
        np.save('atn_trans.npy', atn_trans)
        np.save('atn_em.npy', atn_em)
        print("Building RULE transition and emissions...")
        rule_trans, rule_em = struct_builder.build_rule_hmm_matrices()
        np.save('rule_trans.npy', rule_trans)
        np.save('rule_em.npy', rule_em)
        print("done")
    elif args.tokenize_training_data:
        output_type = args.tokenize_training_data[0]
        DBRunner().tokenize_all_db_source(output_type=output_type)
    elif args.test_atn_hmm_model:
        ATNJavaTokenHMM()
    elif args.test_rule_hmm_model:
        RuleJavaTokenHMM()
    elif args.test_rule_hmm_model_train:
        RuleJavaTokenHMMTrain()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
