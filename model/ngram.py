import kenlm
import os.path


class NGram:
    """
    10-gram model trained on java source code sqlite db.
    Uses token type names.
    """
    ngram_path = os.path.join(os.path.dirname(
        __file__), "java-tokenstr-10grams.arpa")

    def __init__(self):
        if not os.path.isfile(self.ngram_path):
            raise FileNotFoundError("Missing {0}".format(self.ngram_path))
        self.model = kenlm.Model(self.ngram_path)

    def score(self, token_sequence):
        return self.model.score(token_sequence)

    def full_scores(self, token_sequence):
        return self.model.full_scores(token_sequence)

    def perplexity(self, token_sequence):
        return self.model.perplexity(token_sequence)

    def order(self):
        return self.model.order

    def path(self):
        return self.model.path
