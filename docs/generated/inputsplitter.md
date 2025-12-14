## AI Summary

A file named inputsplitter.py.


### Function: num_ini_spaces(s)

**Description:** Return the number of initial spaces in a string.

Note that tabs are counted as a single space.  For now, we do *not* support
mixing of tabs and spaces in the user's input.

Parameters
----------
s : string

Returns
-------
n : int

## Class: IncompleteString

## Class: InMultilineStatement

### Function: partial_tokens(s)

**Description:** Iterate over tokens from a possibly-incomplete string of code.

This adds two special token types: INCOMPLETE_STRING and
IN_MULTILINE_STATEMENT. These can only occur as the last token yielded, and
represent the two main ways for code to be incomplete.

### Function: find_next_indent(code)

**Description:** Find the number of spaces for the next line of indentation

### Function: last_blank(src)

**Description:** Determine if the input source ends in a blank.

A blank is either a newline or a line consisting of whitespace.

Parameters
----------
src : string
    A single or multiline string.

### Function: last_two_blanks(src)

**Description:** Determine if the input source ends in two blanks.

A blank is either a newline or a line consisting of whitespace.

Parameters
----------
src : string
    A single or multiline string.

### Function: remove_comments(src)

**Description:** Remove all comments from input source.

Note: comments are NOT recognized inside of strings!

Parameters
----------
src : string
    A single or multiline input string.

Returns
-------
String with all Python comments removed.

### Function: get_input_encoding()

**Description:** Return the default standard input encoding.

If sys.stdin has no encoding, 'ascii' is returned.

## Class: InputSplitter

**Description:** An object that can accumulate lines of Python source before execution.

This object is designed to be fed python source line-by-line, using
:meth:`push`. It will return on each push whether the currently pushed
code could be executed already. In addition, it provides a method called
:meth:`push_accepts_more` that can be used to query whether more input
can be pushed into a single interactive block.

This is a simple example of how an interactive terminal-based client can use
this tool::

    isp = InputSplitter()
    while isp.push_accepts_more():
        indent = ' '*isp.indent_spaces
        prompt = '>>> ' + indent
        line = indent + raw_input(prompt)
        isp.push(line)
    print('Input source was:\n', isp.source_reset())

## Class: IPythonInputSplitter

**Description:** An input splitter that recognizes all of IPython's special syntax.

### Function: __init__(self, s, start, end, line)

### Function: __init__(self, pos, line)

### Function: _add_indent(n)

### Function: __init__(self)

**Description:** Create a new InputSplitter instance.

### Function: reset(self)

**Description:** Reset the input buffer and associated state.

### Function: source_reset(self)

**Description:** Return the input source and perform a full reset.
        

### Function: check_complete(self, source)

**Description:** Return whether a block of code is ready to execute, or should be continued

This is a non-stateful API, and will reset the state of this InputSplitter.

Parameters
----------
source : string
    Python input code, which can be multiline.

Returns
-------
status : str
    One of 'complete', 'incomplete', or 'invalid' if source is not a
    prefix of valid code.
indent_spaces : int or None
    The number of spaces by which to indent the next line of code. If
    status is not 'incomplete', this is None.

### Function: push(self, lines)

**Description:** Push one or more lines of input.

This stores the given lines and returns a status code indicating
whether the code forms a complete Python block or not.

Any exceptions generated in compilation are swallowed, but if an
exception was produced, the method returns True.

Parameters
----------
lines : string
    One or more lines of Python input.

Returns
-------
is_complete : boolean
    True if the current input source (the result of the current input
    plus prior inputs) forms a complete Python execution block.  Note that
    this value is also stored as a private attribute (``_is_complete``), so it
    can be queried at any time.

### Function: push_accepts_more(self)

**Description:** Return whether a block of interactive input can accept more input.

This method is meant to be used by line-oriented frontends, who need to
guess whether a block is complete or not based solely on prior and
current input lines.  The InputSplitter considers it has a complete
interactive block and will not accept more input when either:

* A SyntaxError is raised

* The code is complete and consists of a single line or a single
  non-compound statement

* The code is complete and has a blank line at the end

If the current input produces a syntax error, this method immediately
returns False but does *not* raise the syntax error exception, as
typically clients will want to send invalid syntax to an execution
backend which might convert the invalid syntax into valid Python via
one of the dynamic IPython mechanisms.

### Function: get_indent_spaces(self)

### Function: _store(self, lines, buffer, store)

**Description:** Store one or more lines of input.

If input lines are not newline-terminated, a newline is automatically
appended.

### Function: _set_source(self, buffer)

### Function: __init__(self, line_input_checker, physical_line_transforms, logical_line_transforms, python_line_transforms)

### Function: transforms(self)

**Description:** Quick access to all transformers.

### Function: transforms_in_use(self)

**Description:** Transformers, excluding logical line transformers if we're in a
Python line.

### Function: reset(self)

**Description:** Reset the input buffer and associated state.

### Function: flush_transformers(self)

### Function: raw_reset(self)

**Description:** Return raw input only and perform a full reset.
        

### Function: source_reset(self)

### Function: push_accepts_more(self)

### Function: transform_cell(self, cell)

**Description:** Process and translate a cell of input.
        

### Function: push(self, lines)

**Description:** Push one or more lines of IPython input.

This stores the given lines and returns a status code indicating
whether the code forms a complete Python block or not, after processing
all input lines for special IPython syntax.

Any exceptions generated in compilation are swallowed, but if an
exception was produced, the method returns True.

Parameters
----------
lines : string
    One or more lines of Python input.

Returns
-------
is_complete : boolean
    True if the current input source (the result of the current input
    plus prior inputs) forms a complete Python execution block.  Note that
    this value is also stored as a private attribute (_is_complete), so it
    can be queried at any time.

### Function: _transform_line(self, line)

**Description:** Push a line of input code through the various transformers.

Returns any output from the transformers, or None if a transformer
is accumulating lines.

Sets self.transformer_accumulating as a side effect.

### Function: _flush(transform, outs)

**Description:** yield transformed lines

always strings, never None

transform: the current transform
outs: an iterable of previously transformed inputs.
     Each may be multiline, which will be passed
     one line at a time to transform.

### Function: _accumulating(dbg)
