## AI Summary

A file named inputtransformer2.py.


### Function: leading_empty_lines(lines)

**Description:** Remove leading empty lines

If the leading lines are empty or contain only whitespace, they will be
removed.

### Function: leading_indent(lines)

**Description:** Remove leading indentation.

If the first line starts with a spaces or tabs, the same whitespace will be
removed from each following line in the cell.

## Class: PromptStripper

**Description:** Remove matching input prompts from a block of input.

Parameters
----------
prompt_re : regular expression
    A regular expression matching any input prompt (including continuation,
    e.g. ``...``)
initial_re : regular expression, optional
    A regular expression matching only the initial prompt, but not continuation.
    If no initial expression is given, prompt_re will be used everywhere.
    Used mainly for plain Python prompts (``>>>``), where the continuation prompt
    ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

Notes
-----

If initial_re and prompt_re differ,
only initial_re will be tested against the first line.
If any prompt is found on the first two lines,
prompts will be stripped from the rest of the block.

### Function: cell_magic(lines)

### Function: _find_assign_op(token_line)

**Description:** Get the index of the first assignment in the line ('=' not inside brackets)

Note: We don't try to support multiple special assignment (a = b = %foo)

### Function: find_end_of_continued_line(lines, start_line)

**Description:** Find the last line of a line explicitly extended using backslashes.

Uses 0-indexed line numbers.

### Function: assemble_continued_line(lines, start, end_line)

**Description:** Assemble a single line from multiple continued line pieces

Continued lines are lines ending in ``\``, and the line following the last
``\`` in the block.

For example, this code continues over multiple lines::

    if (assign_ix is not None) \
         and (len(line) >= assign_ix + 2) \
         and (line[assign_ix+1].string == '%') \
         and (line[assign_ix+2].type == tokenize.NAME):

This statement contains four continued line pieces.
Assembling these pieces into a single line would give::

    if (assign_ix is not None) and (len(line) >= assign_ix + 2) and (line[...

This uses 0-indexed line numbers. *start* is (lineno, colno).

Used to allow ``%magic`` and ``!system`` commands to be continued over
multiple lines.

## Class: TokenTransformBase

**Description:** Base class for transformations which examine tokens.

Special syntax should not be transformed when it occurs inside strings or
comments. This is hard to reliably avoid with regexes. The solution is to
tokenise the code as Python, and recognise the special syntax in the tokens.

IPython's special syntax is not valid Python syntax, so tokenising may go
wrong after the special syntax starts. These classes therefore find and
transform *one* instance of special syntax at a time into regular Python
syntax. After each transformation, tokens are regenerated to find the next
piece of special syntax.

Subclasses need to implement one class method (find)
and one regular method (transform).

The priority attribute can select which transformation to apply if multiple
transformers match in the same place. Lower numbers have higher priority.
This allows "%magic?" to be turned into a help call rather than a magic call.

## Class: MagicAssign

**Description:** Transformer for assignments from magics (a = %foo)

## Class: SystemAssign

**Description:** Transformer for assignments from system commands (a = !foo)

### Function: _make_help_call(target, esc)

**Description:** Prepares a pinfo(2)/psearch call from a target name and the escape
(i.e. ? or ??)

### Function: _tr_help(content)

**Description:** Translate lines escaped with: ?

A naked help line should fire the intro help screen (shell.show_usage())

### Function: _tr_help2(content)

**Description:** Translate lines escaped with: ??

A naked help line should fire the intro help screen (shell.show_usage())

### Function: _tr_magic(content)

**Description:** Translate lines escaped with a percent sign: %

### Function: _tr_quote(content)

**Description:** Translate lines escaped with a comma: ,

### Function: _tr_quote2(content)

**Description:** Translate lines escaped with a semicolon: ;

### Function: _tr_paren(content)

**Description:** Translate lines escaped with a slash: /

## Class: EscapedCommand

**Description:** Transformer for escaped commands like %foo, !foo, or /foo

## Class: HelpEnd

**Description:** Transformer for help syntax: obj? and obj??

### Function: make_tokens_by_line(lines)

**Description:** Tokenize a series of lines and group tokens by line.

The tokens for a multiline Python string or expression are grouped as one
line. All lines except the last lines should keep their line ending ('\n',
'\r\n') for this to properly work. Use `.splitlines(keeplineending=True)`
for example when passing block of text to this function.

### Function: has_sunken_brackets(tokens)

**Description:** Check if the depth of brackets in the list of tokens drops below 0

### Function: show_linewise_tokens(s)

**Description:** For investigation and debugging

## Class: TransformerManager

**Description:** Applies various transformations to a cell or code block.

The key methods for external use are ``transform_cell()``
and ``check_complete()``.

### Function: find_last_indent(lines)

## Class: MaybeAsyncCompile

## Class: MaybeAsyncCommandCompiler

### Function: __init__(self, prompt_re, initial_re)

### Function: _strip(self, lines)

### Function: __call__(self, lines)

### Function: sortby(self)

### Function: __init__(self, start)

### Function: find(cls, tokens_by_line)

**Description:** Find one instance of special syntax in the provided tokens.

Tokens are grouped into logical lines for convenience,
so it is easy to e.g. look at the first token of each line.
*tokens_by_line* is a list of lists of tokenize.TokenInfo objects.

This should return an instance of its class, pointing to the start
position it has found, or None if it found no match.

### Function: transform(self, lines)

**Description:** Transform one instance of special syntax found by ``find()``

Takes a list of strings representing physical lines,
returns a similar list of transformed lines.

### Function: find(cls, tokens_by_line)

**Description:** Find the first magic assignment (a = %foo) in the cell.
        

### Function: transform(self, lines)

**Description:** Transform a magic assignment found by the ``find()`` classmethod.
        

### Function: find_pre_312(cls, tokens_by_line)

### Function: find_post_312(cls, tokens_by_line)

### Function: find(cls, tokens_by_line)

**Description:** Find the first system assignment (a = !foo) in the cell.

### Function: transform(self, lines)

**Description:** Transform a system assignment found by the ``find()`` classmethod.
        

### Function: find(cls, tokens_by_line)

**Description:** Find the first escaped command (%foo, !foo, etc.) in the cell.
        

### Function: transform(self, lines)

**Description:** Transform an escaped line found by the ``find()`` classmethod.
        

### Function: __init__(self, start, q_locn)

### Function: find(cls, tokens_by_line)

**Description:** Find the first help command (foo?) in the cell.
        

### Function: transform(self, lines)

**Description:** Transform a help command found by the ``find()`` classmethod.
        

### Function: __init__(self)

### Function: do_one_token_transform(self, lines)

**Description:** Find and run the transform earliest in the code.

Returns (changed, lines).

This method is called repeatedly until changed is False, indicating
that all available transformations are complete.

The tokens following IPython special syntax might not be valid, so
the transformed code is retokenised every time to identify the next
piece of special syntax. Hopefully long code cells are mostly valid
Python, not using lots of IPython special syntax, so this shouldn't be
a performance issue.

### Function: do_token_transforms(self, lines)

### Function: transform_cell(self, cell)

**Description:** Transforms a cell of input code

### Function: check_complete(self, cell)

**Description:** Return whether a block of code is ready to execute, or should be continued

Parameters
----------
cell : string
    Python input code, which can be multiline.

Returns
-------
status : str
    One of 'complete', 'incomplete', or 'invalid' if source is not a
    prefix of valid code.
indent_spaces : int or None
    The number of spaces by which to indent the next line of code. If
    status is not 'incomplete', this is None.

### Function: __init__(self, extra_flags)

### Function: __init__(self, extra_flags)
