import kenlm
import os.path


class NGram:
    ngram_path = os.path.join(os.path.dirname(__file__), "java-10grams.arpa")

    def __init__(self):
        if not os.path.isfile(self.ngram_path):
            raise FileNotFoundError("Missing {0}".format(self.ngram_path))
        self.model = kenlm.Model(self.ngram_path)

    def score(self, token_sequence):
        return self.model.score(token_sequence, )
