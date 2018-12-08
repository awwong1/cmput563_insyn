from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from grammar.JavaParserListener import JavaParserListener

class CountedDefaultErrorStrategy(DefaultErrorStrategy):
    def __init__(self, *args, **kwargs):
        self.num_errors = 0
        super().__init__(*args, **kwargs)

    def beginErrorCondition(self, recognizer):
        self.num_errors += 1
        super().beginErrorCondition(recognizer)
    
    def reset(self, recognizer):
        self.num_errors = 0
        super().reset(recognizer)


class ParseTreeStepper(JavaParserListener):
    """
    Tree stepper class for printing rules, hacked on from
    https://www.antlr.org/api/Java/org/antlr/v4/runtime/tree/ParseTreeListener.html
    """
    OPEN_RULE = "{"
    CLOSE_RULE = "}"
    SPACER = "  "

    def __init__(self, verbose, **kwargs):
        self.verbose = verbose
        self.literal_node_accum = []
        self.symbolic_node_accum = []

        # get a sequence of antlr parsed text and their corresponding rule
        self._rule_stack = []
        self.text_rule_sequence = [] # tuple of (text, rule)

        super().__init__(**kwargs)

    def enterEveryRule(self, ctx):
        """override"""
        self.symbolic_node_accum.append(
            self.OPEN_RULE + "<R:" + str(ctx.getRuleIndex()) + ">"
        )
        self.literal_node_accum.append(
            self.OPEN_RULE + "<R:" + type(ctx).__name__ + ">"
        )
        self._rule_stack.append(type(ctx).__name__)

    def exitEveryRule(self, ctx):
        """override"""
        self.symbolic_node_accum.append(self.CLOSE_RULE)
        self.literal_node_accum.append(self.CLOSE_RULE)
        self._rule_stack.pop()

    def visitTerminal(self, node):
        """override"""
        self.symbolic_node_accum.append("<T:" + str(node.symbol.type) + ">")
        self.literal_node_accum.append("<T: " + node.symbol.text + ">")

        self.text_rule_sequence.append((node.symbol.text, self._rule_stack[-1],))

    def show_sequence(self):
        """Print out the rule stack"""
        if self.verbose:
            depth = 0
            for text in self.literal_node_accum:
                if self.OPEN_RULE in text:
                    print((self.SPACER * depth) + text)
                    depth += 1
                elif self.CLOSE_RULE in text:
                    depth -= 1
                    print((self.SPACER * depth) + text)
                else:
                    print((self.SPACER * depth) + text)
        else:
            print(" ".join(self.symbolic_node_accum))

    def get_literal_rule_sequence(self):
        """return the literal-rule mapping"""
        return list(self._rule_stack)
