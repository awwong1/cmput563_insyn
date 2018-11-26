import kenlm

class NGram:
    def __init__(self):
        self.model = kenlm.Model('model/test.arpa')
        print(self.model.score('this is a sentence .', bos=True, eos=True))
