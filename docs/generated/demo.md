## AI Summary

A file named demo.py.


## Class: DemoError

### Function: re_mark(mark)

## Class: Demo

## Class: IPythonDemo

**Description:** Class for interactive demos with IPython's input processing applied.

This subclasses Demo, but instead of executing each block by the Python
interpreter (via exec), it actually calls IPython on it, so that any input
filters which may be in place are applied to the input block.

If you have an interactive environment which exposes special input
processing, you can use this class instead to write demo scripts which
operate exactly as if you had typed them interactively.  The default Demo
class requires the input to be valid, pure Python code.

## Class: LineDemo

**Description:** Demo where each line is executed as a separate block.

The input script should be valid Python code.

This class doesn't require any markup at all, and it's meant for simple
scripts (with no nesting or any kind of indentation) which consist of
multiple lines of input to be executed, one at a time, as if they had been
typed in the interactive prompt.

Note: the input can not have *any* indentation, which means that only
single-lines of input are accepted, not even function definitions are
valid.

## Class: IPythonLineDemo

**Description:** Variant of the LineDemo class whose input is processed by IPython.

## Class: ClearMixin

**Description:** Use this mixin to make Demo classes with less visual clutter.

Demos using this mixin will clear the screen before every block and use
blank marquees.

Note that in order for the methods defined here to actually override those
of the classes it's mixed with, it must go /first/ in the inheritance
tree.  For example:

    class ClearIPDemo(ClearMixin,IPythonDemo): pass

will provide an IPythonDemo class with the mixin's features.

## Class: ClearDemo

## Class: ClearIPDemo

### Function: slide(file_path, noclear, format_rst, formatter, style, auto_all, delimiter)

### Function: __init__(self, src, title, arg_str, auto_all, format_rst, formatter, style)

**Description:** Make a new demo object.  To run the demo, simply call the object.

See the module docstring for full details and an example (you can use
IPython.Demo? in IPython to see it).

Inputs:

  - src is either a file, or file-like object, or a
      string that can be resolved to a filename.

Optional inputs:

  - title: a string to use as the demo name.  Of most use when the demo
    you are making comes from an object that has no filename, or if you
    want an alternate denotation distinct from the filename.

  - arg_str(''): a string of arguments, internally converted to a list
    just like sys.argv, so the demo script can see a similar
    environment.

  - auto_all(None): global flag to run all blocks automatically without
    confirmation.  This attribute overrides the block-level tags and
    applies to the whole demo.  It is an attribute of the object, and
    can be changed at runtime simply by reassigning it to a boolean
    value.

  - format_rst(False): a bool to enable comments and doc strings
    formatting with pygments rst lexer

  - formatter('terminal'): a string of pygments formatter name to be
    used. Useful values for terminals: terminal, terminal256,
    terminal16m

  - style('default'): a string of pygments style name to be used.

### Function: fload(self)

**Description:** Load file object.

### Function: reload(self)

**Description:** Reload source from disk and initialize state.

### Function: reset(self)

**Description:** Reset the namespace and seek pointer to restart the demo

### Function: _validate_index(self, index)

### Function: _get_index(self, index)

**Description:** Get the current block index, validating and checking status.

Returns None if the demo is finished

### Function: seek(self, index)

**Description:** Move the current seek pointer to the given block.

You can use negative indices to seek from the end, with identical
semantics to those of Python lists.

### Function: back(self, num)

**Description:** Move the seek pointer back num blocks (default is 1).

### Function: jump(self, num)

**Description:** Jump a given number of blocks relative to the current one.

The offset can be positive or negative, defaults to 1.

### Function: again(self)

**Description:** Move the seek pointer back one block and re-execute.

### Function: edit(self, index)

**Description:** Edit a block.

If no number is given, use the last block executed.

This edits the in-memory copy of the demo, it does NOT modify the
original source file.  If you want to do that, simply open the file in
an editor and use reload() when you make changes to the file.  This
method is meant to let you change a block during a demonstration for
explanatory purposes, without damaging your original script.

### Function: show(self, index)

**Description:** Show a single block on screen

### Function: show_all(self)

**Description:** Show entire demo on screen, block by block

### Function: run_cell(self, source)

**Description:** Execute a string with one or more lines of code

### Function: __call__(self, index)

**Description:** run a block of the demo.

If index is given, it should be an integer >=1 and <= nblocks.  This
means that the calling convention is one off from typical Python
lists.  The reason for the inconsistency is that the demo always
prints 'Block n/N, and N is the total, so it would be very odd to use
zero-indexing here.

### Function: marquee(self, txt, width, mark)

**Description:** Return the input string centered in a 'marquee'.

### Function: pre_cmd(self)

**Description:** Method called before executing each block.

### Function: post_cmd(self)

**Description:** Method called after executing each block.

### Function: highlight(self, block)

**Description:** Method called on each block to highlight it content

### Function: run_cell(self, source)

**Description:** Execute a string with one or more lines of code

### Function: reload(self)

**Description:** Reload source from disk and initialize state.

### Function: marquee(self, txt, width, mark)

**Description:** Blank marquee that returns '' no matter what the input.

### Function: pre_cmd(self)

**Description:** Method called before executing each block.

This one simply clears the screen.
