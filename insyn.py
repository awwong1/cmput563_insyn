#!/usr/bin/env python3
"""INSYN: Reccomendation Models for Syntactically Incorrect Source Code
University of Alberta, CMPUT 563 Fall 2018
"""
import logging
import numpy as np
from argparse import ArgumentParser, SUPPRESS

from analyze.db_runner import DBRunner
from analyze.parser import SourceCodeParser
from analyze.model_tester import ModelTester
from grammar.structure import StructureBuilder
from model.hmm_pom import RuleJavaTokenHMMTrain, TrainedJavaTokenHMM, LabelTrainedJavaTokenHMM


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
        "--evaluate-all-models",
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
    parser.add_argument(
        "--gen-npy-from-training-data",
        help="create a train_data_size_*.npy file from sqlite db",
        type=int,
        metavar="size",
        action="store"
    )
    parser.add_argument(
        "--gen-h5-from-training-data",
        help="create h5 for tokens, corresponding h5 for rules",
        action="store_true"
    )
    parser.add_argument(
        "--train-hmm",
        help="train an hmm with the specified number of components",
        metavar="num_components",
        type=int,
        action="store"
    )

    args = parser.parse_args()
    if args.log:
        raw_log_level = str(args.log).upper()
        if hasattr(logging, raw_log_level):
            log_level = getattr(logging, raw_log_level)
            logging.basicConfig(level=log_level)
            # logging.basicConfig(filename="sample_out.log", level=log_level)
        else:
            logging.warning("Invalid log level: {0}".format(args.log))
    else:
        logging.basicConfig()

    if hasattr(args, "sample_parse"):
        DBRunner().view_one_db_source(args.sample_parse or 0)
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
    elif args.evaluate_all_models:
        ModelTester.init_models()
        ModelTester(args.evaluate_all_models).run_evaluation()
    elif args.test_rule_hmm_model_train:
        RuleJavaTokenHMMTrain()
    elif args.gen_npy_from_training_data:
        DBRunner().create_npy_train_data(args.gen_npy_from_training_data)
    elif args.gen_h5_from_training_data:
        DBRunner().tokenize_with_rule_db_sources()
    elif args.train_hmm:
        # TrainedJavaTokenHMM(args.train_hmm)
        LabelTrainedJavaTokenHMM(args.train_hmm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
