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
        self.literal_rule_stack = []
        self.symbolic_rule_stack = []
        super().__init__(**kwargs)

    def enterEveryRule(self, ctx):
        """override"""
        self.symbolic_rule_stack.append(
            self.OPEN_RULE + "<R:" + str(ctx.getRuleIndex()) + ">"
        )
        self.literal_rule_stack.append(
            self.OPEN_RULE + "<R:" + type(ctx).__name__ + ">"
        )

    def exitEveryRule(self, ctx):
        """override"""
        self.symbolic_rule_stack.append(self.CLOSE_RULE)
        self.literal_rule_stack.append(self.CLOSE_RULE)

    def visitTerminal(self, node):
        """override"""
        self.symbolic_rule_stack.append("<T:" + str(node.symbol.type) + ">")
        self.literal_rule_stack.append("<T: " + node.symbol.text + ">")

    def show_sequence(self):
        """Print out the rule stack"""
        if self.verbose:
            depth = 0
            for text in self.literal_rule_stack:
                if self.OPEN_RULE in text:
                    print((self.SPACER * depth) + text)
                    depth += 1
                elif self.CLOSE_RULE in text:
                    depth -= 1
                    print((self.SPACER * depth) + text)
                else:
                    print((self.SPACER * depth) + text)
        else:
            print(" ".join(self.symbolic_rule_stack))
