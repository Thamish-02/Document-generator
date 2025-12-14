## AI Summary

A file named ansitowin32.py.


## Class: StreamWrapper

**Description:** Wraps a stream (such as stdout), acting as a transparent proxy for all
attribute access apart from method 'write()', which is delegated to our
Converter instance.

## Class: AnsiToWin32

**Description:** Implements a 'write()' method which, on Windows, will strip ANSI character
sequences from the text, and if outputting to a tty, will convert them into
win32 function calls.

### Function: __init__(self, wrapped, converter)

### Function: __getattr__(self, name)

### Function: __enter__(self)

### Function: __exit__(self)

### Function: __setstate__(self, state)

### Function: __getstate__(self)

### Function: write(self, text)

### Function: isatty(self)

### Function: closed(self)

### Function: __init__(self, wrapped, convert, strip, autoreset)

### Function: should_wrap(self)

**Description:** True if this class is actually needed. If false, then the output
stream will not be affected, nor will win32 calls be issued, so
wrapping stdout is not actually required. This will generally be
False on non-Windows platforms, unless optional functionality like
autoreset has been requested using kwargs to init()

### Function: get_win32_calls(self)

### Function: write(self, text)

### Function: reset_all(self)

### Function: write_and_convert(self, text)

**Description:** Write the given text to our wrapped stream, stripping any ANSI
sequences from the text, and optionally converting them into win32
calls.

### Function: write_plain_text(self, text, start, end)

### Function: convert_ansi(self, paramstring, command)

### Function: extract_params(self, command, paramstring)

### Function: call_win32(self, command, params)

### Function: convert_osc(self, text)

### Function: flush(self)
