from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from grammar.JavaParserListener import JavaParserListener
from grammar.JavaParser import JavaParser

# from analyze.parser import SourceCodeParser # throws ImportError (probably circular import..)
JAVA_TOKEN_TYPE_MAP = {
    # "ERROR": -1,  # ERROR,
    "EOF": 0,  # EOF,
    "IDENTIFIER": 1,  # IDENTIFIER(Tokens.Token.Tag.NAMED),
    "ABSTRACT": 2,  # ABSTRACT("abstract"),
    "ASSERT": 3,  # ASSERT("assert", Tokens.Token.Tag.NAMED),
    "BOOLEAN": 4,  # BOOLEAN("boolean", Tokens.Token.Tag.NAMED),
    "BREAK": 5,  # BREAK("break"),
    "BYTE": 6,  # BYTE("byte", Tokens.Token.Tag.NAMED),
    "CASE": 7,  # CASE("case"),
    "CATCH": 8,  # CATCH("catch"),
    "CHAR": 9,  # CHAR("char", Tokens.Token.Tag.NAMED),
    "CLASS": 10,  # CLASS("class"),
    "CONST": 11,  # CONST("const"),
    "CONTINUE": 12,  # CONTINUE("continue"),
    "DEFAULT": 13,  # DEFAULT("default"),
    "DO": 14,  # DO("do"),
    "DOUBLE": 15,  # DOUBLE("double", Tokens.Token.Tag.NAMED),
    "ELSE": 16,  # ELSE("else"),
    "ENUM": 17,  # ENUM("enum", Tokens.Token.Tag.NAMED),
    "EXTENDS": 18,  # EXTENDS("extends"),
    "FINAL": 19,  # FINAL("final"),
    "FINALLY": 20,  # FINALLY("finally"),
    "FLOAT": 21,  # FLOAT("float", Tokens.Token.Tag.NAMED),
    "FOR": 22,  # FOR("for"),
    "GOTO": 23,  # GOTO("goto"),
    "IF": 24,  # IF("if"),
    "IMPLEMENTS": 25,  # IMPLEMENTS("implements"),
    "IMPORT": 26,  # IMPORT("import"),
    "INSTANCEOF": 27,  # INSTANCEOF("instanceof"),
    "INT": 28,  # INT("int", Tokens.Token.Tag.NAMED),
    "INTERFACE": 29,  # INTERFACE("interface"),
    "LONG": 30,  # LONG("long", Tokens.Token.Tag.NAMED),
    "NATIVE": 31,  # NATIVE("native"),
    "NEW": 32,  # NEW("new"),
    "PACKAGE": 33,  # PACKAGE("package"),
    "PRIVATE": 34,  # PRIVATE("private"),
    "PROTECTED": 35,  # PROTECTED("protected"),
    "PUBLIC": 36,  # PUBLIC("public"),
    "RETURN": 37,  # RETURN("return"),
    "SHORT": 38,  # SHORT("short", Tokens.Token.Tag.NAMED),
    "STATIC": 39,  # STATIC("static"),
    "STRICTFP": 40,  # STRICTFP("strictfp"),
    "SUPER": 41,  # SUPER("super", Tokens.Token.Tag.NAMED),
    "SWITCH": 42,  # SWITCH("switch"),
    "SYNCHRONIZED": 43,  # SYNCHRONIZED("synchronized"),
    "THIS": 44,  # THIS("this", Tokens.Token.Tag.NAMED),
    "THROW": 45,  # THROW("throw"),
    "THROWS": 46,  # THROWS("throws"),
    "TRANSIENT": 47,  # TRANSIENT("transient"),
    "TRY": 48,  # TRY("try"),
    "VOID": 49,  # VOID("void", Tokens.Token.Tag.NAMED),
    "VOLATILE": 50,  # VOLATILE("volatile"),
    "WHILE": 51,  # WHILE("while"),
    "INTLITERAL": 52,  # INTLITERAL(Tokens.Token.Tag.NUMERIC),
    "LONGLITERAL": 53,  # LONGLITERAL(Tokens.Token.Tag.NUMERIC),
    "FLOATLITERAL": 54,  # FLOATLITERAL(Tokens.Token.Tag.NUMERIC),
    "DOUBLELITERAL": 55,  # DOUBLELITERAL(Tokens.Token.Tag.NUMERIC),
    "CHARLITERAL": 56,  # CHARLITERAL(Tokens.Token.Tag.NUMERIC),
    "STRINGLITERAL": 57,  # STRINGLITERAL(Tokens.Token.Tag.STRING),
    "TRUE": 58,  # TRUE("true", Tokens.Token.Tag.NAMED),
    "FALSE": 59,  # FALSE("false", Tokens.Token.Tag.NAMED),
    "NULL": 60,  # NULL("null", Tokens.Token.Tag.NAMED),
    "UNDERSCORE": 61,  # UNDERSCORE("_", Tokens.Token.Tag.NAMED),
    "ARROW": 62,  # ARROW("->"),
    "COLCOL": 63,  # COLCOL("::"),
    "LPAREN": 64,  # LPAREN("("),
    "RPAREN": 65,  # RPAREN(")"),
    "LBRACE": 66,  # LBRACE("{"),
    "RBRACE": 67,  # RBRACE("}"),
    "LBRACKET": 68,  # LBRACKET("["),
    "RBRACKET": 69,  # RBRACKET("]"),
    "SEMI": 70,  # SEMI(";"),
    "COMMA": 71,  # COMMA(","),
    "DOT": 72,  # DOT("."),
    "ELLIPSIS": 73,  # ELLIPSIS("..."),
    "EQ": 74,  # EQ("="),
    "GT": 75,  # GT(">"),
    "LT": 76,  # LT("<"),
    "BANG": 77,  # BANG("!"),
    "TILDE": 78,  # TILDE("~"),
    "QUES": 79,  # QUES("?"),
    "COLON": 80,  # COLON(":"),
    "EQEQ": 81,  # EQEQ("=="),
    "LTEQ": 82,  # LTEQ("<="),
    "GTEQ": 83,  # GTEQ(">="),
    "BANGEQ": 84,  # BANGEQ("!="),
    "AMPAMP": 85,  # AMPAMP("&&"),
    "BARBAR": 86,  # BARBAR("||"),
    "PLUSPLUS": 87,  # PLUSPLUS("++"),
    "SUBSUB": 88,  # SUBSUB("--"),
    "PLUS": 89,  # PLUS("+"),
    "SUB": 90,  # SUB("-"),
    "STAR": 91,  # STAR("*"),
    "SLASH": 92,  # SLASH("/"),
    "AMP": 93,  # AMP("&"),
    "BAR": 94,  # BAR("|"),
    "CARET": 95,  # CARET("^"),
    "PERCENT": 96,  # PERCENT("%"),
    "LTLT": 97,  # LTLT("<<"),
    "GTGT": 98,  # GTGT(">>"),
    "GTGTGT": 99,  # GTGTGT(">>>"),
    "PLUSEQ": 100,  # PLUSEQ("+="),
    "SUBEQ": 101,  # SUBEQ("-="),
    "STAREQ": 102,  # STAREQ("*="),
    "SLASHEQ": 103,  # SLASHEQ("/="),
    "AMPEQ": 104,  # AMPEQ("&="),
    "BAREQ": 105,  # BAREQ("|="),
    "CARETEQ": 106,  # CARETEQ("^="),
    "PERCENTEQ": 107,  # PERCENTEQ("%="),
    "LTLTEQ": 108,  # LTLTEQ("<<="),
    "GTGTEQ": 109,  # GTGTEQ(">>="),
    "GTGTGTEQ": 110,  # GTGTGTEQ(">>>="),
    "MONKEYS_AT": 111,  # MONKEYS_AT("@"),
    # "CUSTOM": 112,  # CUSTOM;
}

RULE_NAME_MAPING = list(
    map(lambda x: str(x).upper() + "CONTEXT", JavaParser.ruleNames))


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
        self.text_rule_sequence = []  # tuple of (text, rule)

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

        self.text_rule_sequence.append(
            (node.symbol.text, self._rule_stack[-1],))

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
        return list(self.text_rule_sequence)

    @staticmethod
    def _map_token_rule_to_ints(tokenrule):
        token, rule = tokenrule
        t_id = JAVA_TOKEN_TYPE_MAP.get(token, -1)
        r_id = RULE_NAME_MAPING.index(str(rule).upper())
        return (t_id, r_id)

    def unclean_map_javac_to_antlr(self, javac_token_literal):
        """
        merge ('TOKEN_ID', 'literal value'), ('literal value', 'RULE_NAME') arrays
        this is NOT one to one
        """
        copy_javac_tl = list(javac_token_literal)
        copy_antlr_lt = list(self.text_rule_sequence)

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

        # return javac_token_to_antlr_rule
        # CONVERT TO INTS
        return list(map(lambda x: ParseTreeStepper._map_token_rule_to_ints(x), javac_token_to_antlr_rule))
