## AI Summary

A file named checkers.py.


### Function: num_plurals(catalog, message)

**Description:** Verify the number of plurals in the translation.

### Function: python_format(catalog, message)

**Description:** Verify the format string placeholders in the translation.

### Function: _validate_format(format, alternative)

**Description:** Test format string `alternative` against `format`.  `format` can be the
msgid of a message and `alternative` one of the `msgstr`\s.  The two
arguments are not interchangeable as `alternative` may contain less
placeholders if `format` uses named placeholders.

If the string formatting of `alternative` is compatible to `format` the
function returns `None`, otherwise a `TranslationError` is raised.

Examples for compatible format strings:

>>> _validate_format('Hello %s!', 'Hallo %s!')
>>> _validate_format('Hello %i!', 'Hallo %d!')

Example for an incompatible format strings:

>>> _validate_format('Hello %(name)s!', 'Hallo %s!')
Traceback (most recent call last):
  ...
TranslationError: the format strings are of different kinds

This function is used by the `python_format` checker.

:param format: The original format string
:param alternative: The alternative format string that should be checked
                    against format
:raises TranslationError: on formatting errors

### Function: _find_checkers()

### Function: _parse(string)

### Function: _compatible(a, b)

### Function: _check_positional(results)
