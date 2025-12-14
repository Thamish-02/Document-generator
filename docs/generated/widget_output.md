## AI Summary

A file named widget_output.py.


## Class: Output

**Description:** Widget used as a context manager to display output.

This widget can capture and display stdout, stderr, and rich output.  To use
it, create an instance of it and display it.

You can then use the widget as a context manager: any output produced while in the
context will be captured and displayed in the widget instead of the standard output
area.

You can also use the .capture() method to decorate a function or a method. Any output
produced by the function will then go to the output widget. This is useful for
debugging widget callbacks, for example.

Example::
    import ipywidgets as widgets
    from IPython.display import display
    out = widgets.Output()
    display(out)

    print('prints to output area')

    with out:
        print('prints to output widget')

    @out.capture()
    def func():
        print('prints to output widget')

### Function: clear_output(self)

**Description:** Clear the content of the output widget.

Parameters
----------

wait: bool
    If True, wait to clear the output until new output is
    available to replace it. Default: False

### Function: capture(self, clear_output)

**Description:** Decorator to capture the stdout and stderr of a function.

Parameters
----------

clear_output: bool
    If True, clear the content of the output widget at every
    new function call. Default: False

wait: bool
    If True, wait to clear the output until new output is
    available to replace it. This is only used if clear_output
    is also True.
    Default: False

### Function: __enter__(self)

**Description:** Called upon entering output widget context manager.

### Function: __exit__(self, etype, evalue, tb)

**Description:** Called upon exiting output widget context manager.

### Function: _flush(self)

**Description:** Flush stdout and stderr buffers.

### Function: _append_stream_output(self, text, stream_name)

**Description:** Append a stream output.

### Function: append_stdout(self, text)

**Description:** Append text to the stdout stream.

### Function: append_stderr(self, text)

**Description:** Append text to the stderr stream.

### Function: append_display_data(self, display_object)

**Description:** Append a display object as an output.

Parameters
----------
display_object : IPython.core.display.DisplayObject
    The object to display (e.g., an instance of
    `IPython.display.Markdown` or `IPython.display.Image`).

### Function: capture_decorator(func)

### Function: inner()
