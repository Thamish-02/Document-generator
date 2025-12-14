## AI Summary

A file named highlight.py.


## Class: Highlight2HTML

**Description:** Convert highlighted code to html.

## Class: Highlight2Latex

**Description:** Convert highlighted code to latex.

### Function: _pygments_highlight(source, output_formatter, language, metadata)

**Description:** Return a syntax-highlighted version of the input source

Parameters
----------
source : str
    source of the cell to highlight
output_formatter : Pygments formatter
language : str
    language to highlight the syntax of
metadata : NotebookNode cell metadata
    metadata of the cell to highlight
lexer_options : dict
    Options to pass to the pygments lexer. See
    https://pygments.org/docs/lexers/#available-lexers for more information about
    valid lexer options

### Function: __init__(self, pygments_lexer)

**Description:** Initialize the converter.

### Function: _default_language_changed(self, change)

### Function: __call__(self, source, language, metadata)

**Description:** Return a syntax-highlighted version of the input source as html output.

Parameters
----------
source : str
    source of the cell to highlight
language : str
    language to highlight the syntax of
metadata : NotebookNode cell metadata
    metadata of the cell to highlight

### Function: __init__(self, pygments_lexer)

**Description:** Initialize the converter.

### Function: _default_language_changed(self, change)

### Function: __call__(self, source, language, metadata, strip_verbatim)

**Description:** Return a syntax-highlighted version of the input source as latex output.

Parameters
----------
source : str
    source of the cell to highlight
language : str
    language to highlight the syntax of
metadata : NotebookNode cell metadata
    metadata of the cell to highlight
strip_verbatim : bool
    remove the Verbatim environment that pygments provides by default
