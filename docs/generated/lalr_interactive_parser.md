## AI Summary

A file named lalr_interactive_parser.py.


## Class: InteractiveParser

**Description:** InteractiveParser gives you advanced control over parsing and error handling when parsing with LALR.

For a simpler interface, see the ``on_error`` argument to ``Lark.parse()``.

## Class: ImmutableInteractiveParser

**Description:** Same as ``InteractiveParser``, but operations create a new instance instead
of changing it in-place.

### Function: __init__(self, parser, parser_state, lexer_thread)

### Function: lexer_state(self)

### Function: feed_token(self, token)

**Description:** Feed the parser with a token, and advance it to the next state, as if it received it from the lexer.

Note that ``token`` has to be an instance of ``Token``.

### Function: iter_parse(self)

**Description:** Step through the different stages of the parse, by reading tokens from the lexer
and feeding them to the parser, one per iteration.

Returns an iterator of the tokens it encounters.

When the parse is over, the resulting tree can be found in ``InteractiveParser.result``.

### Function: exhaust_lexer(self)

**Description:** Try to feed the rest of the lexer state into the interactive parser.

Note that this modifies the instance in place and does not feed an '$END' Token

### Function: feed_eof(self, last_token)

**Description:** Feed a '$END' Token. Borrows from 'last_token' if given.

### Function: __copy__(self)

**Description:** Create a new interactive parser with a separate state.

Calls to feed_token() won't affect the old instance, and vice-versa.

### Function: copy(self, deepcopy_values)

### Function: __eq__(self, other)

### Function: as_immutable(self)

**Description:** Convert to an ``ImmutableInteractiveParser``.

### Function: pretty(self)

**Description:** Print the output of ``choices()`` in a way that's easier to read.

### Function: choices(self)

**Description:** Returns a dictionary of token types, matched to their action in the parser.

Only returns token types that are accepted by the current state.

Updated by ``feed_token()``.

### Function: accepts(self)

**Description:** Returns the set of possible tokens that will advance the parser into a new valid state.

### Function: resume_parse(self)

**Description:** Resume automated parsing from the current state.
        

### Function: __hash__(self)

### Function: feed_token(self, token)

### Function: exhaust_lexer(self)

**Description:** Try to feed the rest of the lexer state into the parser.

Note that this returns a new ImmutableInteractiveParser and does not feed an '$END' Token

### Function: as_mutable(self)

**Description:** Convert to an ``InteractiveParser``.
