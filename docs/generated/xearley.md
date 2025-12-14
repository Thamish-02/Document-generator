## AI Summary

A file named xearley.py.


## Class: Parser

### Function: __init__(self, lexer_conf, parser_conf, term_matcher, resolve_ambiguity, complete_lex, debug, tree_class, ordered_sets)

### Function: _parse(self, stream, columns, to_scan, start_symbol)

### Function: scan(i, to_scan)

**Description:** The core Earley Scanner.

This is a custom implementation of the scanner that uses the
Lark lexer to match tokens. The scan list is built by the
Earley predictor, based on the previously completed tokens.
This ensures that at each phase of the parse we have a custom
lexer context, allowing for more complex ambiguities.
