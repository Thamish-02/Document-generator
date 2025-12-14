## AI Summary

A file named ansi.py.


### Function: strip_ansi(source)

**Description:** Remove ANSI escape codes from text.

Parameters
----------
source : str
    Source to remove the ANSI from

### Function: ansi2html(text)

**Description:** Convert ANSI colors to HTML colors.

Parameters
----------
text : unicode
    Text containing ANSI colors to convert to HTML

### Function: ansi2latex(text)

**Description:** Convert ANSI colors to LaTeX colors.

Parameters
----------
text : unicode
    Text containing ANSI colors to convert to LaTeX

### Function: _htmlconverter(fg, bg, bold, underline, inverse)

**Description:** Return start and end tags for given foreground/background/bold/underline.

### Function: _latexconverter(fg, bg, bold, underline, inverse)

**Description:** Return start and end markup given foreground/background/bold/underline.

### Function: _ansi2anything(text, converter)

**Description:** Convert ANSI colors to HTML or LaTeX.

See https://en.wikipedia.org/wiki/ANSI_escape_code

Accepts codes like '\x1b[32m' (red) and '\x1b[1;32m' (bold, red).

Non-color escape sequences (not ending with 'm') are filtered out.

Ideally, this should have the same behavior as the function
fixConsole() in notebook/notebook/static/base/js/utils.js.

### Function: _get_extended_color(numbers)
