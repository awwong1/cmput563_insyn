#!/usr/bin/env python
from model.hmm_pom import Trained10StateHMM, Trained100StateHMM, ATNJavaTokenHMM
from model.ngram import KenLM10Gram
from analyze.parser import SourceCodeParser
from antlr4 import ParseTreeWalker

from grammar.tree_helper import ParseTreeStepper

TEST_SRC = """
public class HelloWorld {
    private static TestObject<Foo<Bar>> testobject;
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
"""


def unclean_map_javac_to_antlr(javac_token_literal, antlr_literal_rule):
    """
    merge ('TOKEN_ID', 'literal value'), ('literal value', 'RULE_NAME') arrays
    this is NOT one to one
    """
    copy_javac_tl = list(javac_token_literal)
    copy_antlr_lt = list(antlr_literal_rule)

    copy_javac_tl.reverse()
    copy_antlr_lt.reverse()

    javac_token_to_antlr_rule = []
    literal_partial = ""

    while len(copy_javac_tl) and len(copy_antlr_lt):
        (javac_token, javac_literal) = copy_javac_tl.pop()
        (antlr_literal, antlr_rule) = copy_antlr_lt.pop()
        # print(javac_literal, antlr_literal, javac_literal == antlr_literal)
        if javac_literal == antlr_literal:
            # great, base case, we done
            javac_token_to_antlr_rule.append((javac_token, antlr_rule,))
            literal_partial = ""
        elif javac_literal == "" and antlr_literal == "<EOF>":
            javac_token_to_antlr_rule.append((javac_token, antlr_rule,))
            literal_partial = ""
        elif javac_literal == literal_partial + antlr_literal:
            # constructed literals are okay too
            javac_token_to_antlr_rule.append((javac_token, antlr_rule))
            literal_partial = ""
        else:
            # stupid ">" ">>" cases
            literal_partial += antlr_literal
            copy_javac_tl.append((javac_token, javac_literal,))

    return javac_token_to_antlr_rule


(javac_num_errors, javac_token_sequence) = SourceCodeParser().javac_analyze(TEST_SRC)

# hmm wants [0, 1, 2, ..., 111]
input_for_hmm = list(SourceCodeParser.tokens_to_ints(javac_token_sequence))
# ngram wants ["PACKAGE", "IDENTIFIER", ..., "EOF"]
input_for_ngram = list(map(lambda x: x[0], javac_token_sequence))

print("======= SOURCE INPUT =======")
print(TEST_SRC)
print("======= ANTLR TOKENS =======")
(antlr_num_errors, antlr_tokens, tree) = SourceCodeParser.antlr_analyze(TEST_SRC)
print("ANTLR NUM ERRORS FOUND: {}".format(antlr_num_errors))
str_antlr_tokens = map(
    lambda antlr_token: (
        # tree.parser.symbolicNames[antlr_token.type]
        antlr_token.text
    ), antlr_tokens)
# print(" ".join(list(str_antlr_tokens)))
printer = ParseTreeStepper(False)
walker = ParseTreeWalker()
walker.walk(printer, tree)
antlr_literal_to_rule = printer.get_literal_rule_sequence()
# print(antlr_literal_to_rule)

print("======= JAVAC TOKENS =======")
print("JAVAC NUM ERRORS FOUND: {}".format(javac_num_errors))
# print(list(zip(input_for_ngram, input_for_hmm)))
# print(" ".join(input_for_ngram))
javac_token_to_literal = list(
    map(lambda x: (x[0], x[1]), javac_token_sequence))
# print(javac_token_to_literal)

print("======= UNCLEAN MAP =======")
token_to_rule = unclean_map_javac_to_antlr(javac_token_to_literal, antlr_literal_to_rule)

print(token_to_rule)

print("======= MODEL EVAL =======")
MODELS = {
    # "10-gram": KenLM10Gram(),
    # "10-hmm": Trained10StateHMM(),
    # "100-hmm": Trained100StateHMM(),
    # "atn-hmm": ATNJavaTokenHMM()
}
SCORES = {
    "10-gram": 0,
    "10-hmm": 0,
    "100-hmm": 0,
    "atn-hmm": 0
}

header_1 = "seq\t| \t\t|"
header_2 = "idx\t| token\t\t|"
header_3 = "--------|---------------|"
for model_name in MODELS.keys():
    header_1 += " sum(logprob)\t| {} token\t|".format(model_name)
    header_2 += " {}\t| delta logprob\t|".format(model_name)
    header_3 += "---------------|---------------|"

print(header_1)
print(header_2)
print(header_3)

for idx in range(1, len(javac_token_sequence) + 1):
    row = "{idx}\t| {token:{fill}{align}{pad}}\t".format(
        idx=idx, token=input_for_ngram[idx-1], fill=" ", align="<", pad=9)
    for model_name, model in MODELS.items():
        seq_score = 0
        if model_name == "10-gram":
            seq_score = model.score(" ".join(input_for_ngram[:idx]))
        else:
            seq_score = model.score(input_for_hmm[:idx])
        delta = seq_score - SCORES[model_name]
        SCORES[model_name] = seq_score
        row += "| {seq_score:.3f}\t| {delta:.4f}\t".format(
            seq_score=seq_score,
            delta=delta
        )
    print(row)
