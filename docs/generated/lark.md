## AI Summary

A file named lark.py.


## Class: PostLex

## Class: LarkOptions

**Description:** Specifies the options for Lark

    

## Class: Lark

**Description:** Main interface for the library.

It's mostly a thin wrapper for the many different parsers, and for the tree constructor.

Parameters:
    grammar: a string or file-object containing the grammar spec (using Lark's ebnf syntax)
    options: a dictionary controlling various aspects of Lark.

Example:
    >>> Lark(r'''start: "foo" ''')
    Lark(...)

### Function: process(self, stream)

### Function: __init__(self, options_dict)

### Function: __getattr__(self, name)

### Function: __setattr__(self, name, value)

### Function: serialize(self, memo)

### Function: deserialize(cls, data, memo)

### Function: __init__(self, grammar)

### Function: _build_lexer(self, dont_ignore)

### Function: _prepare_callbacks(self)

### Function: _build_parser(self)

### Function: save(self, f, exclude_options)

**Description:** Saves the instance into the given file object

Useful for caching and multiprocessing.

### Function: load(cls, f)

**Description:** Loads an instance from the given file object

Useful for caching and multiprocessing.

### Function: _deserialize_lexer_conf(self, data, memo, options)

### Function: _load(self, f)

### Function: _load_from_dict(cls, data, memo)

### Function: open(cls, grammar_filename, rel_to)

**Description:** Create an instance of Lark with the grammar given by its filename

If ``rel_to`` is provided, the function will find the grammar filename in relation to it.

Example:

    >>> Lark.open("grammar_file.lark", rel_to=__file__, parser="lalr")
    Lark(...)

### Function: open_from_package(cls, package, grammar_path, search_paths)

**Description:** Create an instance of Lark with the grammar loaded from within the package `package`.
This allows grammar loading from zipapps.

Imports in the grammar will use the `package` and `search_paths` provided, through `FromPackageLoader`

Example:

    Lark.open_from_package(__name__, "example.lark", ("grammars",), parser=...)

### Function: __repr__(self)

### Function: lex(self, text, dont_ignore)

**Description:** Only lex (and postlex) the text, without parsing it. Only relevant when lexer='basic'

When dont_ignore=True, the lexer will return all tokens, even those marked for %ignore.

:raises UnexpectedCharacters: In case the lexer cannot find a suitable match.

### Function: get_terminal(self, name)

**Description:** Get information about a terminal

### Function: parse_interactive(self, text, start)

**Description:** Start an interactive parsing session. Only works when parser='lalr'.

Parameters:
    text (LarkInput, optional): Text to be parsed. Required for ``resume_parse()``.
    start (str, optional): Start symbol

Returns:
    A new InteractiveParser instance.

See Also: ``Lark.parse()``

### Function: parse(self, text, start, on_error)

**Description:** Parse the given text, according to the options provided.

Parameters:
    text (LarkInput): Text to be parsed, as `str` or `bytes`.
        TextSlice may also be used, but only when lexer='basic' or 'contextual'.
        If Lark was created with a custom lexer, this may be an object of any type.
    start (str, optional): Required if Lark was given multiple possible start symbols (using the start option).
    on_error (function, optional): if provided, will be called on UnexpectedInput error,
        with the exception as its argument. Return true to resume parsing, or false to raise the exception.
        LALR only. See examples/advanced/error_handling.py for an example of how to use on_error.

Returns:
    If a transformer is supplied to ``__init__``, returns whatever is the
    result of the transformation. Otherwise, returns a Tree instance.

:raises UnexpectedInput: On a parse error, one of these sub-exceptions will rise:
        ``UnexpectedCharacters``, ``UnexpectedToken``, or ``UnexpectedEOF``.
        For convenience, these sub-exceptions also inherit from ``ParserError`` and ``LexerError``.
