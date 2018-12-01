// Modified JavaLexer to work with javac outputted tokens
/*
 [The "BSD licence"]
 Copyright (c) 2013 Terence Parr, Sam Harwell
 Copyright (c) 2017 Ivan Kochurkin (upgrade to Java 8)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

lexer grammar JavaLexer;

// Keywords

ABSTRACT:           'abstract';
ASSERT:             'assert';
BOOLEAN:            'boolean';
BREAK:              'break';
BYTE:               'byte';
CASE:               'case';
CATCH:              'catch';
CHAR:               'char';
CLASS:              'class';
CONST:              'const';
CONTINUE:           'continue';
DEFAULT:            'default';
DO:                 'do';
DOUBLE:             'double';
ELSE:               'else';
ENUM:               'enum';
EXTENDS:            'extends';
FINAL:              'final';
FINALLY:            'finally';
FLOAT:              'float';
FOR:                'for';
IF:                 'if';
GOTO:               'goto';
IMPLEMENTS:         'implements';
IMPORT:             'import';
INSTANCEOF:         'instanceof';
INT:                'int';
INTERFACE:          'interface';
LONG:               'long';
NATIVE:             'native';
NEW:                'new';
PACKAGE:            'package';
PRIVATE:            'private';
PROTECTED:          'protected';
PUBLIC:             'public';
RETURN:             'return';
SHORT:              'short';
STATIC:             'static';
STRICTFP:           'strictfp';
SUPER:              'super';
SWITCH:             'switch';
SYNCHRONIZED:       'synchronized';
THIS:               'this';
THROW:              'throw';
THROWS:             'throws';
TRANSIENT:          'transient';
TRY:                'try';
VOID:               'void';
VOLATILE:           'volatile';
WHILE:              'while';

// Literals
INTLITERAL: ('0' | [1-9] (Digits? | '_'+ Digits))              // decimal literal
          | '0' [xX] [0-9a-fA-F] ([0-9a-fA-F_]* [0-9a-fA-F])?  // hex literal
          | '0' '_'* [0-7] ([0-7_]* [0-7])?                    // oct literal
          | '0' [bB] [01] ([01_]* [01])?                       // binary literal
          ;

LONGLITERAL: ('0' | [1-9] (Digits? | '_'+ Digits)) [lL]
           | '0' [xX] [0-9a-fA-F] ([0-9a-fA-F_]* [0-9a-fA-F])? [lL]
           | '0' '_'* [0-7] ([0-7_]* [0-7])? [lL]
           | '0' [bB] [01] ([01_]* [01])? [lL]
           ;

FLOATLITERAL: (Digits '.' Digits? | '.' Digits) ExponentPart? [fF]
            | Digits (ExponentPart [fF] | [fF])
            | '0' [xX] (HexDigits '.'? | HexDigits? '.' HexDigits) [pP] [+-]? Digits [fF]
            ;

DOUBLELITERAL: (Digits '.' Digits? | '.' Digits) ExponentPart? [dD]?
            | Digits (ExponentPart [dD]? | [dD]?)
            | '0' [xX] (HexDigits '.'? | HexDigits? '.' HexDigits) [pP] [+-]? Digits [dD]?
            ;


TRUE: 'true';
FALSE: 'false';

CHARLITERAL:       '\'' (~['\\\r\n] | EscapeSequence) '\'';

STRINGLITERAL:     '"' (~["\\\r\n] | EscapeSequence)* '"';

NULL:       'null';

// Separators

LPAREN:             '(';
RPAREN:             ')';
LBRACE:             '{';
RBRACE:             '}';
LBRACKET:             '[';
RBRACKET:             ']';
SEMI:               ';';
COMMA:              ',';
DOT:                '.';

// Operators

EQ:                 '=';
GT:                 '>';
LT:                 '<';
BANG:               '!';
TILDE:              '~';
QUES:           '?';
COLON:              ':';
EQEQ:              '==';
LTEQ:                 '<=';
GTEQ:                 '>=';
BANGEQ:           '!=';
AMPAMP:                '&&';
BARBAR:                 '||';
PLUSPLUS:                '++';
SUBSUB:                '--';
PLUS:                '+';
SUB:                '-';
STAR:                '*';
SLASH:                '/';
AMP:             '&';
BAR:              '|';
CARET:              '^';
PERCENT:                '%';

LTLT: '<<';
GTGT: '>>';
GTGTGT: '>>>';

PLUSEQ:         '+=';
SUBEQ:         '-=';
STAREQ:         '*=';
SLASHEQ:         '/=';
AMPEQ:         '&=';
BAREQ:          '|=';
CARETEQ:         '^=';
PERCENTEQ:         '%=';
LTLTEQ:      '<<=';
GTGTEQ:      '>>=';
GTGTGTEQ:     '>>>=';

// Java 8 tokens

ARROW:              '->';
COLCOL:         '::';

// Additional symbols not defined in the lexical specification

MONKEYS_AT:                 '@';
ELLIPSIS:           '...';

// Identifiers
UNDERSCORE:         '_';
IDENTIFIER:         Letter LetterOrDigit*;

// Whitespace and comments

WS:                 [ \t\r\n\u000C]+ -> channel(HIDDEN);
COMMENT:            '/*' .*? '*/'    -> channel(HIDDEN);
LINE_COMMENT:       '//' ~[\r\n]*    -> channel(HIDDEN);

// Fragment rules

fragment ExponentPart
    : [eE] [+-]? Digits
    ;

fragment EscapeSequence
    : '\\' [btnfr"'\\]
    | '\\' ([0-3]? [0-7])? [0-7]
    | '\\' 'u'+ HexDigit HexDigit HexDigit HexDigit
    ;

fragment HexDigits
    : HexDigit ((HexDigit | '_')* HexDigit)?
    ;

fragment HexDigit
    : [0-9a-fA-F]
    ;

fragment Digits
    : [0-9] ([0-9_]* [0-9])?
    ;

fragment LetterOrDigit
    : Letter
    | [0-9]
    ;

fragment Letter
    : [a-zA-Z$_] // these are the "java letters" below 0x7F
    | ~[\u0000-\u007F\uD800-\uDBFF] // covers all characters above 0x7F which are not a surrogate
    | [\uD800-\uDBFF] [\uDC00-\uDFFF] // covers UTF-16 surrogate pairs encodings for U+10000 to U+10FFFF
    ;
