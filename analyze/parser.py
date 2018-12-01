import time
from antlr4 import InputStream, CommonTokenStream
from javac_parser import Java

from grammar.JavaParser import JavaParser
from grammar.JavaLexer import JavaLexer
from grammar.tree_helper import CountedDefaultErrorStrategy, ParseTreeStepper


class SourceCodeParser:
    # from javac TokenKind enum
    JAVA_TOKEN_TYPE_MAP = {
        # "ERROR": -1,  # ERROR,
        "EOF": 1,  # EOF,
        "IDENTIFIER": 2,  # IDENTIFIER(Tokens.Token.Tag.NAMED),
        "ABSTRACT": 3,  # ABSTRACT("abstract"),
        "ASSERT": 4,  # ASSERT("assert", Tokens.Token.Tag.NAMED),
        "BOOLEAN": 5,  # BOOLEAN("boolean", Tokens.Token.Tag.NAMED),
        "BREAK": 6,  # BREAK("break"),
        "BYTE": 7,  # BYTE("byte", Tokens.Token.Tag.NAMED),
        "CASE": 8,  # CASE("case"),
        "CATCH": 9,  # CATCH("catch"),
        "CHAR": 10,  # CHAR("char", Tokens.Token.Tag.NAMED),
        "CLASS": 11,  # CLASS("class"),
        "CONST": 12,  # CONST("const"),
        "CONTINUE": 13,  # CONTINUE("continue"),
        "DEFAULT": 14,  # DEFAULT("default"),
        "DO": 15,  # DO("do"),
        "DOUBLE": 16,  # DOUBLE("double", Tokens.Token.Tag.NAMED),
        "ELSE": 17,  # ELSE("else"),
        "ENUM": 18,  # ENUM("enum", Tokens.Token.Tag.NAMED),
        "EXTENDS": 19,  # EXTENDS("extends"),
        "FINAL": 20,  # FINAL("final"),
        "FINALLY": 21,  # FINALLY("finally"),
        "FLOAT": 22,  # FLOAT("float", Tokens.Token.Tag.NAMED),
        "FOR": 23,  # FOR("for"),
        "GOTO": 24,  # GOTO("goto"),
        "IF": 25,  # IF("if"),
        "IMPLEMENTS": 26,  # IMPLEMENTS("implements"),
        "IMPORT": 27,  # IMPORT("import"),
        "INSTANCEOF": 28,  # INSTANCEOF("instanceof"),
        "INT": 29,  # INT("int", Tokens.Token.Tag.NAMED),
        "INTERFACE": 30,  # INTERFACE("interface"),
        "LONG": 31,  # LONG("long", Tokens.Token.Tag.NAMED),
        "NATIVE": 32,  # NATIVE("native"),
        "NEW": 33,  # NEW("new"),
        "PACKAGE": 34,  # PACKAGE("package"),
        "PRIVATE": 35,  # PRIVATE("private"),
        "PROTECTED": 36,  # PROTECTED("protected"),
        "PUBLIC": 37,  # PUBLIC("public"),
        "RETURN": 38,  # RETURN("return"),
        "SHORT": 39,  # SHORT("short", Tokens.Token.Tag.NAMED),
        "STATIC": 40,  # STATIC("static"),
        "STRICTFP": 41,  # STRICTFP("strictfp"),
        "SUPER": 42,  # SUPER("super", Tokens.Token.Tag.NAMED),
        "SWITCH": 43,  # SWITCH("switch"),
        "SYNCHRONIZED": 44,  # SYNCHRONIZED("synchronized"),
        "THIS": 45,  # THIS("this", Tokens.Token.Tag.NAMED),
        "THROW": 46,  # THROW("throw"),
        "THROWS": 47,  # THROWS("throws"),
        "TRANSIENT": 48,  # TRANSIENT("transient"),
        "TRY": 49,  # TRY("try"),
        "VOID": 50,  # VOID("void", Tokens.Token.Tag.NAMED),
        "VOLATILE": 51,  # VOLATILE("volatile"),
        "WHILE": 52,  # WHILE("while"),
        "INTLITERAL": 53,  # INTLITERAL(Tokens.Token.Tag.NUMERIC),
        "LONGLITERAL": 54,  # LONGLITERAL(Tokens.Token.Tag.NUMERIC),
        "FLOATLITERAL": 55,  # FLOATLITERAL(Tokens.Token.Tag.NUMERIC),
        "DOUBLELITERAL": 56,  # DOUBLELITERAL(Tokens.Token.Tag.NUMERIC),
        "CHARLITERAL": 57,  # CHARLITERAL(Tokens.Token.Tag.NUMERIC),
        "STRINGLITERAL": 58,  # STRINGLITERAL(Tokens.Token.Tag.STRING),
        "TRUE": 59,  # TRUE("true", Tokens.Token.Tag.NAMED),
        "FALSE": 60,  # FALSE("false", Tokens.Token.Tag.NAMED),
        "NULL": 61,  # NULL("null", Tokens.Token.Tag.NAMED),
        "UNDERSCORE": 62,  # UNDERSCORE("_", Tokens.Token.Tag.NAMED),
        "ARROW": 63,  # ARROW("->"),
        "COLCOL": 64,  # COLCOL("::"),
        "LPAREN": 65,  # LPAREN("("),
        "RPAREN": 66,  # RPAREN(")"),
        "LBRACE": 67,  # LBRACE("{"),
        "RBRACE": 68,  # RBRACE("}"),
        "LBRACKET": 69,  # LBRACKET("["),
        "RBRACKET": 70,  # RBRACKET("]"),
        "SEMI": 71,  # SEMI(";"),
        "COMMA": 72,  # COMMA(","),
        "DOT": 73,  # DOT("."),
        "ELLIPSIS": 74,  # ELLIPSIS("..."),
        "EQ": 75,  # EQ("="),
        "GT": 76,  # GT(">"),
        "LT": 77,  # LT("<"),
        "BANG": 78,  # BANG("!"),
        "TILDE": 79,  # TILDE("~"),
        "QUES": 80,  # QUES("?"),
        "COLON": 81,  # COLON(":"),
        "EQEQ": 82,  # EQEQ("=="),
        "LTEQ": 83,  # LTEQ("<="),
        "GTEQ": 84,  # GTEQ(">="),
        "BANGEQ": 85,  # BANGEQ("!="),
        "AMPAMP": 86,  # AMPAMP("&&"),
        "BARBAR": 87,  # BARBAR("||"),
        "PLUSPLUS": 88,  # PLUSPLUS("++"),
        "SUBSUB": 89,  # SUBSUB("--"),
        "PLUS": 90,  # PLUS("+"),
        "SUB": 91,  # SUB("-"),
        "STAR": 92,  # STAR("*"),
        "SLASH": 93,  # SLASH("/"),
        "AMP": 94,  # AMP("&"),
        "BAR": 95,  # BAR("|"),
        "CARET": 96,  # CARET("^"),
        "PERCENT": 97,  # PERCENT("%"),
        "LTLT": 98,  # LTLT("<<"),
        "GTGT": 99,  # GTGT(">>"),
        "GTGTGT": 100,  # GTGTGT(">>>"),
        "PLUSEQ": 101,  # PLUSEQ("+="),
        "SUBEQ": 102,  # SUBEQ("-="),
        "STAREQ": 103,  # STAREQ("*="),
        "SLASHEQ": 104,  # SLASHEQ("/="),
        "AMPEQ": 105,  # AMPEQ("&="),
        "BAREQ": 106,  # BAREQ("|="),
        "CARETEQ": 107,  # CARETEQ("^="),
        "PERCENTEQ": 108,  # PERCENTEQ("%="),
        "LTLTEQ": 109,  # LTLTEQ("<<="),
        "GTGTEQ": 110,  # GTGTEQ(">>="),
        "GTGTGTEQ": 111,  # GTGTGTEQ(">>>="),
        "MONKEYS_AT": 112,  # MONKEYS_AT("@"),
        "CUSTOM": 113,  # CUSTOM;
    }
    JAVA_TOKEN_ID_MAP = {
        v: k for k, v in JAVA_TOKEN_TYPE_MAP.items()
    }

    def __init__(self):
        self.javac = Java() # each thread must have its own instance of Javac

    @staticmethod
    def tokens_to_ints(tuple_tokens):
        """
        convert javac token_sequence into interable of integers
        """
        if tuple_tokens:
            str_tokens = tuple_tokens
            if len(tuple_tokens[0]) > 1:
                str_tokens = map(lambda tup: tup[0], tuple_tokens)
            return map(
                lambda str_token: SourceCodeParser.JAVA_TOKEN_TYPE_MAP.get(str_token, -1), str_tokens)

    @staticmethod
    def antlr_analyze(source_code, remove_whitespace=False):
        source_buf = InputStream(source_code)
        lexer = JavaLexer(source_buf)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(input=stream)
        parser._errHandler = CountedDefaultErrorStrategy()

        tree = parser.compilationUnit()
        antlr_tokens = stream.tokens
        if remove_whitespace:
            cleaned_antlr_tokens = []
            for common_token in antlr_tokens:
                symb = parser.symbolicNames[common_token.type]
                if symb not in ["WS", "COMMENT", "LINE_COMMENT"]:
                    cleaned_antlr_tokens.append(common_token)
            antlr_tokens = cleaned_antlr_tokens
        num_errors = parser._errHandler.num_errors

        return (num_errors, antlr_tokens, tree)

    def javac_analyze(self, source_code):
        num_errors = self.javac.get_num_parse_errors(source_code)
        token_sequence = self.javac.lex(source_code)
        return (num_errors, token_sequence)
    
    def javac_check_syntax(self, source_code):
        return self.javac.check_syntax(source_code)
