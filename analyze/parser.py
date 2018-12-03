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
        "CUSTOM": 112,  # CUSTOM;
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
    def antlr_analyze(source_code, remove_whitespace=True):
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
