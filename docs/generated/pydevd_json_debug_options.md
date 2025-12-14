## AI Summary

A file named pydevd_json_debug_options.py.


## Class: DebugOptions

### Function: int_parser(s, default_value)

### Function: bool_parser(s)

### Function: unquote(s)

### Function: _build_debug_options(flags)

**Description:** Build string representation of debug options from the launch config.

### Function: _parse_debug_options(opts)

**Description:** Debug options are semicolon separated key=value pairs

### Function: _extract_debug_options(opts, flags)

**Description:** Return the debug options encoded in the given value.

"opts" is a semicolon-separated string of "key=value" pairs.
"flags" is a list of strings.

If flags is provided then it is used as a fallback.

The values come from the launch config:

 {
     type:'python',
     request:'launch'|'attach',
     name:'friendly name for debug config',
     debugOptions:[
         'RedirectOutput', 'Django'
     ],
     options:'REDIRECT_OUTPUT=True;DJANGO_DEBUG=True'
 }

Further information can be found here:

https://code.visualstudio.com/docs/editor/debugging#_launchjson-attributes

### Function: __init__(self)

### Function: to_json(self)

### Function: update_fom_debug_options(self, debug_options)

### Function: update_from_args(self, args)
