## AI Summary

A file named ultratb.py.


### Function: count_lines_in_py_file(filename)

**Description:** Given a filename, returns the number of lines in the file
if it ends with the extension ".py". Otherwise, returns 0.

### Function: get_line_number_of_frame(frame)

**Description:** Given a frame object, returns the total number of lines in the file
containing the frame's code object, or the number of lines in the
frame's source code if the file is not available.

Parameters
----------
frame : FrameType
    The frame object whose line number is to be determined.

Returns
-------
int
    The total number of lines in the file containing the frame's
    code object, or the number of lines in the frame's source code
    if the file is not available.

### Function: _safe_string(value, what, func)

### Function: _format_traceback_lines(lines, Colors, has_colors, lvals)

**Description:** Format tracebacks lines with pointing arrow, leading numbers...

Parameters
----------
lines : list[Line]
Colors
    ColorScheme used.
lvals : str
    Values of local variables, already colored, to inject just after the error line.

### Function: _simple_format_traceback_lines(lnum, index, lines, Colors, lvals, _line_format)

**Description:** Format tracebacks lines with pointing arrow, leading numbers...

Parameters
==========

lnum: int
    number of the target line of code.
index: int
    which line in the list should be highlighted.
lines: list[string]
Colors:
    ColorScheme used.
lvals: bytes
    Values of local variables, already colored, to inject just after the error line.
_line_format: f (str) -> (str, bool)
    return (colorized version of str, failure to do so)

### Function: _format_filename(file, ColorFilename, ColorNormal)

**Description:** Format filename lines with custom formatting from caching compiler or `File *.py` by default

Parameters
----------
file : str
ColorFilename
    ColorScheme's filename coloring to be used.
ColorNormal
    ColorScheme's normal coloring to be used.

## Class: TBTools

**Description:** Basic tools used by all traceback printer classes.

## Class: ListTB

**Description:** Print traceback information from a traceback list, with optional color.

Calling requires 3 arguments: (etype, evalue, elist)
as would be obtained by::

  etype, evalue, tb = sys.exc_info()
  if tb:
    elist = traceback.extract_tb(tb)
  else:
    elist = None

It can thus be used by programs which need to process the traceback before
printing (such as console replacements based on the code module from the
standard library).

Because they are meant to be called without a full traceback (only a
list), instances of this class can't call the interactive pdb debugger.

## Class: FrameInfo

**Description:** Mirror of stack data's FrameInfo, but so that we can bypass highlighting on
really long frames.

## Class: VerboseTB

**Description:** A port of Ka-Ping Yee's cgitb.py module that outputs color text instead
of HTML.  Requires inspect and pydoc.  Crazy, man.

Modified version which optionally strips the topmost entries from the
traceback, to be used with alternate interpreters (because their own code
would appear in the traceback).

## Class: FormattedTB

**Description:** Subclass ListTB but allow calling with a traceback.

It can thus be used as a sys.excepthook for Python > 2.1.

Also adds 'Context' and 'Verbose' modes, not available in ListTB.

Allows a tb_offset to be specified. This is useful for situations where
one needs to remove a number of topmost frames from the traceback (such as
occurs with python programs that themselves execute other python code,
like Python shells).  

## Class: AutoFormattedTB

**Description:** A traceback printer which can be called on the fly.

It will find out about exceptions by itself.

A brief example::

    AutoTB = AutoFormattedTB(mode = 'Verbose',color_scheme='Linux')
    try:
      ...
    except:
      AutoTB()  # or AutoTB(out=logfile) where logfile is an open file object

## Class: ColorTB

**Description:** Shorthand to initialize a FormattedTB in Linux colors mode.

## Class: SyntaxTB

**Description:** Extension which holds some state: the last exception value

### Function: text_repr(value)

**Description:** Hopefully pretty robust repr equivalent.

### Function: eqrepr(value, repr)

### Function: nullrepr(value, repr)

### Function: __init__(self, color_scheme, call_pdb, ostream, parent, config)

### Function: _get_ostream(self)

**Description:** Output stream that exceptions are written to.

Valid values are:

- None: the default, which means that IPython will dynamically resolve
  to sys.stdout.  This ensures compatibility with most tools, including
  Windows (where plain stdout doesn't recognize ANSI escapes).

- Any object with 'write' and 'flush' attributes.

### Function: _set_ostream(self, val)

### Function: _get_chained_exception(exception_value)

### Function: get_parts_of_chained_exception(self, evalue)

### Function: prepare_chained_exception_message(self, cause)

### Function: has_colors(self)

### Function: set_colors(self)

**Description:** Shorthand access to the color table scheme selector method.

### Function: color_toggle(self)

**Description:** Toggle between the currently active color scheme and NoColor.

### Function: stb2text(self, stb)

**Description:** Convert a structured traceback (a list) to a string.

### Function: text(self, etype, value, tb, tb_offset, context)

**Description:** Return formatted traceback.

Subclasses may override this if they add extra arguments.

### Function: structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)

**Description:** Return a list of traceback frames.

Must be implemented by each class.

### Function: __call__(self, etype, value, elist)

### Function: _extract_tb(self, tb)

### Function: structured_traceback(self, etype, evalue, etb, tb_offset, context)

**Description:** Return a color formatted string with the traceback info.

Parameters
----------
etype : exception type
    Type of the exception raised.
evalue : object
    Data stored in the exception
etb : list | TracebackType | None
    If list: List of frames, see class docstring for details.
    If Traceback: Traceback of the exception.
tb_offset : int, optional
    Number of frames in the traceback to skip.  If not given, the
    instance evalue is used (set in constructor).
context : int, optional
    Number of lines of context information to print.

Returns
-------
String with formatted exception.

### Function: _format_list(self, extracted_list)

**Description:** Format a list of traceback entry tuples for printing.

Given a list of tuples as returned by extract_tb() or
extract_stack(), return a list of strings ready for printing.
Each string in the resulting list corresponds to the item with the
same index in the argument list.  Each string ends in a newline;
the strings may contain internal newlines as well, for those items
whose source text line is not None.

Lifted almost verbatim from traceback.py

### Function: _format_exception_only(self, etype, value)

**Description:** Format the exception part of a traceback.

The arguments are the exception type and value such as given by
sys.exc_info()[:2]. The return value is a list of strings, each ending
in a newline.  Normally, the list contains a single string; however,
for SyntaxError exceptions, it contains several lines that (when
printed) display detailed information about where the syntax error
occurred.  The message indicating which exception occurred is the
always last string in the list.

Also lifted nearly verbatim from traceback.py

### Function: get_exception_only(self, etype, value)

**Description:** Only print the exception type and message, without a traceback.

Parameters
----------
etype : exception type
value : exception value

### Function: show_exception_only(self, etype, evalue)

**Description:** Only print the exception type and message, without a traceback.

Parameters
----------
etype : exception type
evalue : exception value

### Function: _some_str(self, value)

### Function: _from_stack_data_FrameInfo(cls, frame_info)

### Function: __init__(self, description, filename, lineno, frame, code)

### Function: variables_in_executing_piece(self)

### Function: lines(self)

### Function: executing(self)

### Function: __init__(self, color_scheme, call_pdb, ostream, tb_offset, long_header, include_vars, check_cache, debugger_cls, parent, config)

**Description:** Specify traceback offset, headers and color scheme.

Define how many frames to drop from the tracebacks. Calling it with
tb_offset=1 allows use of this handler in interpreters which will have
their own code at the top of the traceback (VerboseTB will first
remove that frame before printing the traceback info).

### Function: format_record(self, frame_info)

**Description:** Format a single stack frame

### Function: prepare_header(self, etype, long_version)

### Function: format_exception(self, etype, evalue)

### Function: format_exception_as_a_whole(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)

**Description:** Formats the header, traceback and exception message for a single exception.

This may be called multiple times by Python 3 exception chaining
(PEP 3134).

### Function: get_records(self, etb, number_of_lines_of_context, tb_offset)

### Function: structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)

**Description:** Return a nice text document describing the traceback.

### Function: debugger(self, force)

**Description:** Call up the pdb debugger if desired, always clean up the tb
reference.

Keywords:

  - force(False): by default, this routine checks the instance call_pdb
    flag and does not actually invoke the debugger if the flag is false.
    The 'force' option forces the debugger to activate even if the flag
    is false.

If the call_pdb flag is set, the pdb interactive debugger is
invoked. In all cases, the self.tb reference to the current traceback
is deleted to prevent lingering references which hamper memory
management.

Note that each call to pdb() does an 'import readline', so if your app
requires a special setup for the readline completers, you'll have to
fix that by hand after invoking the exception handler.

### Function: handler(self, info)

### Function: __call__(self, etype, evalue, etb)

**Description:** This hook can replace sys.excepthook (for Python 2.1 or higher).

### Function: __init__(self, mode, color_scheme, call_pdb, ostream, tb_offset, long_header, include_vars, check_cache, debugger_cls, parent, config)

### Function: structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)

### Function: stb2text(self, stb)

**Description:** Convert a structured traceback (a list) to a string.

### Function: set_mode(self, mode)

**Description:** Switch to the desired mode.

If mode is not specified, cycles through the available modes.

### Function: plain(self)

### Function: context(self)

### Function: verbose(self)

### Function: minimal(self)

### Function: __call__(self, etype, evalue, etb, out, tb_offset)

**Description:** Print out a formatted exception traceback.

Optional arguments:
  - out: an open file-like object to direct output to.

  - tb_offset: the number of frames to skip over in the stack, on a
  per-call basis (this overrides temporarily the instance's tb_offset
  given at initialization time.

### Function: structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)

### Function: __init__(self, color_scheme, call_pdb)

### Function: __init__(self, color_scheme, parent, config)

### Function: __call__(self, etype, value, elist)

### Function: structured_traceback(self, etype, value, elist, tb_offset, context)

### Function: clear_err_state(self)

**Description:** Return the current error state and clear it

### Function: stb2text(self, stb)

**Description:** Convert a structured traceback (a list) to a string.

## Class: Dummy

### Function: render(self)
