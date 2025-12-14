## AI Summary

A file named indenter.py.


## Class: DedentError

## Class: Indenter

**Description:** This is a postlexer that "injects" indent/dedent tokens based on indentation.

It keeps track of the current indentation, as well as the current level of parentheses.
Inside parentheses, the indentation is ignored, and no indent/dedent tokens get generated.

Note: This is an abstract class. To use it, inherit and implement all its abstract methods:
    - tab_len
    - NL_type
    - OPEN_PAREN_types, CLOSE_PAREN_types
    - INDENT_type, DEDENT_type

See also: the ``postlex`` option in `Lark`.

## Class: PythonIndenter

**Description:** A postlexer that "injects" _INDENT/_DEDENT tokens based on indentation, according to the Python syntax.

See also: the ``postlex`` option in `Lark`.

### Function: __init__(self)

### Function: handle_NL(self, token)

### Function: _process(self, stream)

### Function: process(self, stream)

### Function: always_accept(self)

### Function: NL_type(self)

**Description:** The name of the newline token

### Function: OPEN_PAREN_types(self)

**Description:** The names of the tokens that open a parenthesis

### Function: CLOSE_PAREN_types(self)

**Description:** The names of the tokens that close a parenthesis
        

### Function: INDENT_type(self)

**Description:** The name of the token that starts an indentation in the grammar.

See also: %declare

### Function: DEDENT_type(self)

**Description:** The name of the token that end an indentation in the grammar.

See also: %declare

### Function: tab_len(self)

**Description:** How many spaces does a tab equal
