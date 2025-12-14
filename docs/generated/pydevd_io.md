## AI Summary

A file named pydevd_io.py.


## Class: IORedirector

**Description:** This class works to wrap a stream (stdout/stderr) with an additional redirect.

## Class: RedirectToPyDBIoMessages

## Class: IOBuf

**Description:** This class works as a replacement for stdio and stderr.
It is a buffer and when its contents are requested, it will erase what
it has so far so that the next return will not return the same contents again.

## Class: _RedirectInfo

## Class: _RedirectionsHolder

### Function: start_redirect(keep_original_redirection, std, redirect_to)

**Description:** @param std: 'stdout', 'stderr', or 'both'

### Function: end_redirect(std)

### Function: redirect_stream_to_pydb_io_messages(std)

**Description:** :param std:
    'stdout' or 'stderr'

### Function: stop_redirect_stream_to_pydb_io_messages(std)

**Description:** :param std:
    'stdout' or 'stderr'

### Function: redirect_stream_to_pydb_io_messages_context()

### Function: __init__(self, original, new_redirect, wrap_buffer)

**Description:** :param stream original:
    The stream to be wrapped (usually stdout/stderr, but could be None).

:param stream new_redirect:
    Usually IOBuf (below).

:param bool wrap_buffer:
    Whether to create a buffer attribute (needed to mimick python 3 s
    tdout/stderr which has a buffer to write binary data).

### Function: write(self, s)

### Function: isatty(self)

### Function: flush(self)

### Function: __getattr__(self, name)

### Function: __init__(self, out_ctx, wrap_stream, wrap_buffer, on_write)

**Description:** :param out_ctx:
    1=stdout and 2=stderr

:param wrap_stream:
    Either sys.stdout or sys.stderr.

:param bool wrap_buffer:
    If True the buffer attribute (which wraps writing bytes) should be
    wrapped.

:param callable(str) on_write:
    May be a custom callable to be called when to write something.
    If not passed the default implementation will create an io message
    and send it through the debugger.

### Function: get_pydb(self)

### Function: flush(self)

### Function: write(self, s)

### Function: __init__(self)

### Function: getvalue(self)

### Function: write(self, s)

### Function: isatty(self)

### Function: flush(self)

### Function: empty(self)

### Function: __init__(self, original, redirect_to)
