## AI Summary

A file named ptutils.py.


### Function: _elide_point(string)

**Description:** If a string is long enough, and has at least 3 dots,
replace the middle part with ellipses.

If a string naming a file is long enough, and has at least 3 slashes,
replace the middle part with ellipses.

If three consecutive dots, or two consecutive dots are encountered these are
replaced by the equivalents HORIZONTAL ELLIPSIS or TWO DOT LEADER unicode
equivalents

### Function: _elide_typed(string, typed)

**Description:** Elide the middle of a long string if the beginning has already been typed.

### Function: _elide(string, typed, min_elide)

### Function: _adjust_completion_text_based_on_context(text, body, offset)

## Class: IPythonPTCompleter

**Description:** Adaptor to provide IPython completions to prompt_toolkit

## Class: IPythonPTLexer

**Description:** Wrapper around PythonLexer and BashLexer.

### Function: __init__(self, ipy_completer, shell)

### Function: ipy_completer(self)

### Function: get_completions(self, document, complete_event)

### Function: _get_completions(body, offset, cursor_position, ipyc)

**Description:** Private equivalent of get_completions() use only for unit_testing.

### Function: __init__(self)

### Function: lex_document(self, document)
