import kenlm
import os.path


class KenLM10Gram:
    """
    10-gram model trained on java source code sqlite db.
    Uses token type names.
    """
    ngram_path = os.path.join(os.path.dirname(
        __file__), "java-tokenstr-10grams.arpa")

    def __init__(self):
        # check for .arpa.bin, then .arpa, else fail
        if os.path.isfile(self.ngram_path + ".bin"):
            self.model = kenlm.Model(self.ngram_path + ".bin")
        elif os.path.isfile(self.ngram_path):
            self.model = kenlm.Model(self.ngram_path)
        else:
            raise FileNotFoundError("Missing {0}".format(self.ngram_path))

    def score(self, token_sequence):
        return self.model.score(token_sequence)

    def full_scores(self, token_sequence):
        return self.model.full_scores(token_sequence)


class KenLM2Gram:
    """
    2-gram model trained on java source code sqlite db.
    Uses token type names.
    """
    ngram_path = os.path.join(os.path.dirname(
        __file__), "java-tokenstr-2grams.arpa")

    def __init__(self):
        # check for .arpa.bin, then .arpa, else fail
        if os.path.isfile(self.ngram_path + ".bin"):
            self.model = kenlm.Model(self.ngram_path + ".bin")
        elif os.path.isfile(self.ngram_path):
            self.model = kenlm.Model(self.ngram_path)
        else:
            raise FileNotFoundError("Missing {0}".format(self.ngram_path))

    def score(self, token_sequence):
        return self.model.score(token_sequence)

    def full_scores(self, token_sequence):
        return self.model.full_scores(token_sequence)
