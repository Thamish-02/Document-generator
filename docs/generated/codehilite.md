## AI Summary

A file named codehilite.py.


### Function: parse_hl_lines(expr)

**Description:** Support our syntax for emphasizing certain lines of code.

`expr` should be like '1 2' to emphasize lines 1 and 2 of a code block.
Returns a list of integers, the line numbers to emphasize.

## Class: CodeHilite

**Description:** Determine language of source code, and pass it on to the Pygments highlighter.

Usage:

```python
code = CodeHilite(src=some_code, lang='python')
html = code.hilite()
```

Arguments:
    src: Source string or any object with a `.readline` attribute.

Keyword arguments:
    lang (str): String name of Pygments lexer to use for highlighting. Default: `None`.
    guess_lang (bool): Auto-detect which lexer to use.
        Ignored if `lang` is set to a valid value. Default: `True`.
    use_pygments (bool): Pass code to Pygments for code highlighting. If `False`, the code is
        instead wrapped for highlighting by a JavaScript library. Default: `True`.
    pygments_formatter (str): The name of a Pygments formatter or a formatter class used for
        highlighting the code blocks. Default: `html`.
    linenums (bool): An alias to Pygments `linenos` formatter option. Default: `None`.
    css_class (str): An alias to Pygments `cssclass` formatter option. Default: 'codehilite'.
    lang_prefix (str): Prefix prepended to the language. Default: "language-".

Other Options:

Any other options are accepted and passed on to the lexer and formatter. Therefore,
valid options include any options which are accepted by the `html` formatter or
whichever lexer the code's language uses. Note that most lexers do not have any
options. However, a few have very useful options, such as PHP's `startinline` option.
Any invalid options are ignored without error.

* **Formatter options**: <https://pygments.org/docs/formatters/#HtmlFormatter>
* **Lexer Options**: <https://pygments.org/docs/lexers/>

Additionally, when Pygments is enabled, the code's language is passed to the
formatter as an extra option `lang_str`, whose value being `{lang_prefix}{lang}`.
This option has no effect to the Pygments' builtin formatters.

Advanced Usage:

```python
code = CodeHilite(
    src = some_code,
    lang = 'php',
    startinline = True,      # Lexer option. Snippet does not start with `<?php`.
    linenostart = 42,        # Formatter option. Snippet starts on line 42.
    hl_lines = [45, 49, 50], # Formatter option. Highlight lines 45, 49, and 50.
    linenos = 'inline'       # Formatter option. Avoid alignment problems.
)
html = code.hilite()
```

## Class: HiliteTreeprocessor

**Description:** Highlight source code in code blocks. 

## Class: CodeHiliteExtension

**Description:** Add source code highlighting to markdown code blocks. 

### Function: makeExtension()

### Function: __init__(self, src)

### Function: hilite(self, shebang)

**Description:** Pass code to the [Pygments](https://pygments.org/) highlighter with
optional line numbers. The output should then be styled with CSS to
your liking. No styles are applied by default - only styling hooks
(i.e.: `<span class="k">`).

returns : A string of html.

### Function: _parseHeader(self)

**Description:** Determines language of a code block from shebang line and whether the
said line should be removed or left in place. If the shebang line
contains a path (even a single /) then it is assumed to be a real
shebang line and left alone. However, if no path is given
(e.i.: `#!python` or `:::python`) then it is assumed to be a mock shebang
for language identification of a code fragment and removed from the
code block prior to processing for code highlighting. When a mock
shebang (e.i: `#!python`) is found, line numbering is turned on. When
colons are found in place of a shebang (e.i.: `:::python`), line
numbering is left in the current state - off by default.

Also parses optional list of highlight lines, like:

    :::python hl_lines="1 3"

### Function: code_unescape(self, text)

**Description:** Unescape code.

### Function: run(self, root)

**Description:** Find code blocks and store in `htmlStash`. 

### Function: __init__(self)

### Function: extendMarkdown(self, md)

**Description:** Add `HilitePostprocessor` to Markdown instance. 
