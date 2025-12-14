## AI Summary

A file named earley.py.


## Class: Parser

### Function: __init__(self, lexer_conf, parser_conf, term_matcher, resolve_ambiguity, debug, tree_class, ordered_sets)

### Function: predict_and_complete(self, i, to_scan, columns, transitives, node_cache)

**Description:** The core Earley Predictor and Completer.

At each stage of the input, we handling any completed items (things
that matched on the last cycle) and use those to predict what should
come next in the input stream. The completions and any predicted
non-terminals are recursively processed until we reach a set of,
which can be added to the scan list for the next scanner cycle.

### Function: _parse(self, lexer, columns, to_scan, start_symbol)

### Function: parse(self, lexer, start)

### Function: is_quasi_complete(item)

### Function: scan(i, token, to_scan)

**Description:** The core Earley Scanner.

This is a custom implementation of the scanner that uses the
Lark lexer to match tokens. The scan list is built by the
Earley predictor, based on the previously completed tokens.
This ensures that at each phase of the parse we have a custom
lexer context, allowing for more complex ambiguities.
