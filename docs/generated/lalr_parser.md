## AI Summary

A file named lalr_parser.py.


## Class: LALR_Parser

## Class: _Parser

### Function: __init__(self, parser_conf, debug, strict)

### Function: deserialize(cls, data, memo, callbacks, debug)

### Function: serialize(self, memo)

### Function: parse_interactive(self, lexer, start)

### Function: parse(self, lexer, start, on_error)

### Function: __init__(self, parse_table, callbacks, debug)

### Function: parse(self, lexer, start, value_stack, state_stack, start_interactive)

### Function: parse_from_state(self, state, last_token)

**Description:** Run the main LALR parser loop

Parameters:
    state - the initial state. Changed in-place.
    last_token - Used only for line information in case of an empty lexer.
