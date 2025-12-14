## AI Summary

A file named lexer.py.


## Class: Pattern

**Description:** An abstraction over regular expressions.

## Class: PatternStr

## Class: PatternRE

## Class: TerminalDef

**Description:** A definition of a terminal

## Class: Token

**Description:** A string with meta-information, that is produced by the lexer.

When parsing text, the resulting chunks of the input that haven't been discarded,
will end up in the tree as Token instances. The Token class inherits from Python's ``str``,
so normal string comparisons and operations will work as expected.

Attributes:
    type: Name of the token (as specified in grammar)
    value: Value of the token (redundant, as ``token.value == token`` will always be true)
    start_pos: The index of the token in the text
    line: The line of the token in the text (starting with 1)
    column: The column of the token in the text (starting with 1)
    end_line: The line where the token ends
    end_column: The next column after the end of the token. For example,
        if the token is a single character with a column value of 4,
        end_column will be 5.
    end_pos: the index where the token ends (basically ``start_pos + len(token)``)

## Class: LineCounter

**Description:** A utility class for keeping track of line & column information

## Class: UnlessCallback

## Class: CallChain

### Function: _get_match(re_, regexp, s, flags)

### Function: _create_unless(terminals, g_regex_flags, re_, use_bytes)

## Class: Scanner

### Function: _regexp_has_newline(r)

**Description:** Expressions that may indicate newlines in a regexp:
- newlines (\n)
- escaped newline (\\n)
- anything but ([^...])
- any-char (.) when the flag (?s) exists
- spaces (\s)

## Class: LexerState

**Description:** Represents the current state of the lexer as it scans the text
(Lexer objects are only instantiated per grammar, not per text)

## Class: LexerThread

**Description:** A thread that ties a lexer instance and a lexer state, to be used by the parser
    

## Class: Lexer

**Description:** Lexer interface

Method Signatures:
    lex(self, lexer_state, parser_state) -> Iterator[Token]

### Function: _check_regex_collisions(terminal_to_regexp, comparator, strict_mode, max_collisions_to_show)

## Class: AbstractBasicLexer

## Class: BasicLexer

## Class: ContextualLexer

### Function: __init__(self, value, flags, raw)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __eq__(self, other)

### Function: to_regexp(self)

### Function: min_width(self)

### Function: max_width(self)

### Function: _get_flags(self, value)

### Function: to_regexp(self)

### Function: min_width(self)

### Function: max_width(self)

### Function: to_regexp(self)

### Function: _get_width(self)

### Function: min_width(self)

### Function: max_width(self)

### Function: __init__(self, name, pattern, priority)

### Function: __repr__(self)

### Function: user_repr(self)

### Function: __new__(cls, type, value, start_pos, line, column, end_line, end_column, end_pos)

### Function: __new__(cls, type_, value, start_pos, line, column, end_line, end_column, end_pos)

### Function: __new__(cls)

### Function: _future_new(cls, type, value, start_pos, line, column, end_line, end_column, end_pos)

### Function: update(self, type, value)

### Function: update(self, type_, value)

### Function: update(self)

### Function: _future_update(self, type, value)

### Function: new_borrow_pos(cls, type_, value, borrow_t)

### Function: __reduce__(self)

### Function: __repr__(self)

### Function: __deepcopy__(self, memo)

### Function: __eq__(self, other)

### Function: __init__(self, newline_char)

### Function: __eq__(self, other)

### Function: feed(self, token, test_newline)

**Description:** Consume a token and calculate the new line & column.

As an optional optimization, set test_newline=False if token doesn't contain a newline.

### Function: __init__(self, scanner)

### Function: __call__(self, t)

### Function: __init__(self, callback1, callback2, cond)

### Function: __call__(self, t)

### Function: __init__(self, terminals, g_regex_flags, re_, use_bytes)

### Function: _build_mres(self, terminals, max_size)

### Function: match(self, text, pos)

### Function: fullmatch(self, text)

### Function: __init__(self, text, line_ctr, last_token)

### Function: __eq__(self, other)

### Function: __copy__(self)

### Function: __init__(self, lexer, lexer_state)

### Function: from_text(cls, lexer, text_or_slice)

### Function: from_custom_input(cls, lexer, text)

### Function: lex(self, parser_state)

### Function: __copy__(self)

### Function: lex(self, lexer_state, parser_state)

### Function: make_lexer_state(self, text)

**Description:** Deprecated

### Function: __init__(self, conf, comparator)

### Function: next_token(self, lex_state, parser_state)

### Function: lex(self, state, parser_state)

### Function: __init__(self, conf, comparator)

### Function: _build_scanner(self)

### Function: scanner(self)

### Function: match(self, text, pos)

### Function: next_token(self, lex_state, parser_state)

### Function: __init__(self, conf, states, always_accept)

### Function: lex(self, lexer_state, parser_state)
