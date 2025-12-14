## AI Summary

A file named displayhook.py.


## Class: DisplayHook

**Description:** The custom IPython displayhook to replace sys.displayhook.

This class does many things, but the basic idea is that it is a callable
that gets called anytime user code returns a value.

## Class: CapturingDisplayHook

### Function: __init__(self, shell, cache_size)

### Function: prompt_count(self)

### Function: check_for_underscore(self)

**Description:** Check if the user has set the '_' variable by hand.

### Function: quiet(self)

**Description:** Should we silence the display hook because of ';'?

### Function: semicolon_at_end_of_expression(expression)

**Description:** Parse Python expression and detects whether last token is ';'

### Function: start_displayhook(self)

**Description:** Start the displayhook, initializing resources.

### Function: write_output_prompt(self)

**Description:** Write the output prompt.

The default implementation simply writes the prompt to
``sys.stdout``.

### Function: compute_format_data(self, result)

**Description:** Compute format data of the object to be displayed.

The format data is a generalization of the :func:`repr` of an object.
In the default implementation the format data is a :class:`dict` of
key value pair where the keys are valid MIME types and the values
are JSON'able data structure containing the raw data for that MIME
type. It is up to frontends to determine pick a MIME to to use and
display that data in an appropriate manner.

This method only computes the format data for the object and should
NOT actually print or write that to a stream.

Parameters
----------
result : object
    The Python object passed to the display hook, whose format will be
    computed.

Returns
-------
(format_dict, md_dict) : dict
    format_dict is a :class:`dict` whose keys are valid MIME types and values are
    JSON'able raw data for that MIME type. It is recommended that
    all return values of this should always include the "text/plain"
    MIME type representation of the object.
    md_dict is a :class:`dict` with the same MIME type keys
    of metadata associated with each output.

### Function: write_format_data(self, format_dict, md_dict)

**Description:** Write the format data dict to the frontend.

This default version of this method simply writes the plain text
representation of the object to ``sys.stdout``. Subclasses should
override this method to send the entire `format_dict` to the
frontends.

Parameters
----------
format_dict : dict
    The format dict for the object passed to `sys.displayhook`.
md_dict : dict (optional)
    The metadata dict to be associated with the display data.

### Function: update_user_ns(self, result)

**Description:** Update user_ns with various things like _, __, _1, etc.

### Function: fill_exec_result(self, result)

### Function: log_output(self, format_dict)

**Description:** Log the output.

### Function: finish_displayhook(self)

**Description:** Finish up all displayhook activities.

### Function: __call__(self, result)

**Description:** Printing with history cache management.

This is invoked every time the interpreter needs to print, and is
activated by setting the variable sys.displayhook to it.

### Function: cull_cache(self)

**Description:** Output cache is full, cull the oldest entries

### Function: flush(self)

### Function: __init__(self, shell, outputs)

### Function: __call__(self, result)
