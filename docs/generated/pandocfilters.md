## AI Summary

A file named pandocfilters.py.


### Function: get_filename4code(module, content, ext)

**Description:** Generate filename based on content

The function ensures that the (temporary) directory exists, so that the
file can be written.

By default, the directory won't be cleaned up,
so a filter can use the directory as a cache and
decide not to regenerate if there's no change.

In case the user preferres the files to be temporary files,
an environment variable `PANDOCFILTER_CLEANUP` can be set to
any non-empty value such as `1` to
make sure the directory is created in a temporary location and removed
after finishing the filter. In this case there's no caching and files
will be regenerated each time the filter is run.

Example:
    filename = get_filename4code("myfilter", code)

### Function: get_value(kv, key, value)

**Description:** get value from the keyvalues (options)

### Function: get_caption(kv)

**Description:** get caption from the keyvalues (options)

Example:
  if key == 'CodeBlock':
    [[ident, classes, keyvals], code] = value
    caption, typef, keyvals = get_caption(keyvals)
    ...
    return Para([Image([ident, [], keyvals], caption, [filename, typef])])

### Function: get_extension(format, default)

**Description:** get the extension for the result, needs a default and some specialisations

Example:
  filetype = get_extension(format, "png", html="svg", latex="eps")

### Function: walk(x, action, format, meta)

**Description:** Walk a tree, applying an action to every object.
Returns a modified tree.  An action is a function of the form
`action(key, value, format, meta)`, where:

* `key` is the type of the pandoc object (e.g. 'Str', 'Para') `value` is
* the contents of the object (e.g. a string for 'Str', a list of
  inline elements for 'Para')
* `format` is the target output format (as supplied by the
  `format` argument of `walk`)
* `meta` is the document's metadata

The return of an action is either:

* `None`: this means that the object should remain unchanged
* a pandoc object: this will replace the original object
* a list of pandoc objects: these will replace the original object; the
  list is merged with the neighbors of the orignal objects (spliced into
  the list the original object belongs to); returning an empty list deletes
  the object

### Function: toJSONFilter(action)

**Description:** Like `toJSONFilters`, but takes a single action as argument.
    

### Function: toJSONFilters(actions)

**Description:** Generate a JSON-to-JSON filter from stdin to stdout

The filter:

* reads a JSON-formatted pandoc document from stdin
* transforms it by walking the tree and performing the actions
* returns a new JSON-formatted pandoc document to stdout

The argument `actions` is a list of functions of the form
`action(key, value, format, meta)`, as described in more
detail under `walk`.

This function calls `applyJSONFilters`, with the `format`
argument provided by the first command-line argument,
if present.  (Pandoc sets this by default when calling
filters.)

### Function: applyJSONFilters(actions, source, format)

**Description:** Walk through JSON structure and apply filters

This:

* reads a JSON-formatted pandoc document from a source string
* transforms it by walking the tree and performing the actions
* returns a new JSON-formatted pandoc document as a string

The `actions` argument is a list of functions (see `walk`
for a full description).

The argument `source` is a string encoded JSON object.

The argument `format` is a string describing the output format.

Returns a the new JSON-formatted pandoc document.

### Function: stringify(x)

**Description:** Walks the tree x and returns concatenated string content,
leaving out all formatting.

### Function: attributes(attrs)

**Description:** Returns an attribute list, constructed from the
dictionary attrs.

### Function: elt(eltType, numargs)

### Function: go(key, val, format, meta)

### Function: fun()
