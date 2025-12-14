## AI Summary

A file named excolors.py.


### Function: exception_colors()

**Description:** Return a color table with fields for exception reporting.

The table is an instance of ColorSchemeTable with schemes added for
'Neutral', 'Linux', 'LightBG' and 'NoColor' and fields for exception handling filled
in.

Examples:

>>> ec = exception_colors()
>>> ec.active_scheme_name
''
>>> print(ec.active_colors)
None

Now we activate a color scheme:
>>> ec.set_active_scheme('NoColor')
>>> ec.active_scheme_name
'NoColor'
>>> sorted(ec.active_colors.keys())
['Normal', 'breakpoint_disabled', 'breakpoint_enabled', 'caret', 'em',
'excName', 'filename', 'filenameEm', 'line', 'lineno', 'linenoEm', 'name',
'nameEm', 'normalEm', 'prompt', 'topline', 'vName', 'val', 'valEm']
