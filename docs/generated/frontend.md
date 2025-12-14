## AI Summary

A file named frontend.py.


## Class: BaseError

## Class: OptionError

## Class: SetupError

## Class: ConfigurationError

**Description:** Raised for errors in configuration files.

### Function: listify_value(arg, split)

**Description:** Make a list out of an argument.

Values from `distutils` argument parsing are always single strings;
values from `optparse` parsing may be lists of strings that may need
to be further split.

No matter the input, this function returns a flat list of whitespace-trimmed
strings, with `None` values filtered out.

>>> listify_value("foo bar")
['foo', 'bar']
>>> listify_value(["foo bar"])
['foo', 'bar']
>>> listify_value([["foo"], "bar"])
['foo', 'bar']
>>> listify_value([["foo"], ["bar", None, "foo"]])
['foo', 'bar', 'foo']
>>> listify_value("foo, bar, quux", ",")
['foo', 'bar', 'quux']

:param arg: A string or a list of strings
:param split: The argument to pass to `str.split()`.
:return:

## Class: CommandMixin

## Class: CompileCatalog

### Function: _make_directory_filter(ignore_patterns)

**Description:** Build a directory_filter function based on a list of ignore patterns.

## Class: ExtractMessages

## Class: InitCatalog

## Class: UpdateCatalog

## Class: CommandLineInterface

**Description:** Command-line interface.

This class provides a simple command-line interface to the message
extraction and PO file generation functionality.

### Function: main()

### Function: parse_mapping(fileobj, filename)

### Function: parse_mapping_cfg(fileobj, filename)

**Description:** Parse an extraction method mapping from a file-like object.

:param fileobj: a readable file-like object containing the configuration
                text to parse
:param filename: the name of the file being parsed, for error messages

### Function: _parse_config_object(config)

### Function: _parse_mapping_toml(fileobj, filename, style)

**Description:** Parse an extraction method mapping from a binary file-like object.

.. warning: As of this version of Babel, this is a private API subject to changes.

:param fileobj: a readable binary file-like object containing the configuration TOML to parse
:param filename: the name of the file being parsed, for error messages
:param style: whether the file is in the style of a `pyproject.toml` file, i.e. whether to look for `tool.babel`.

### Function: _parse_spec(s)

### Function: parse_keywords(strings)

**Description:** Parse keywords specifications from the given list of strings.

>>> import pprint
>>> keywords = ['_', 'dgettext:2', 'dngettext:2,3', 'pgettext:1c,2',
...             'polymorphic:1', 'polymorphic:2,2t', 'polymorphic:3c,3t']
>>> pprint.pprint(parse_keywords(keywords))
{'_': None,
 'dgettext': (2,),
 'dngettext': (2, 3),
 'pgettext': ((1, 'c'), 2),
 'polymorphic': {None: (1,), 2: (2,), 3: ((3, 'c'),)}}

The input keywords are in GNU Gettext style; see :doc:`cmdline` for details.

The output is a dictionary mapping keyword names to a dictionary of specifications.
Keys in this dictionary are numbers of arguments, where ``None`` means that all numbers
of arguments are matched, and a number means only calls with that number of arguments
are matched (which happens when using the "t" specifier). However, as a special
case for backwards compatibility, if the dictionary of specifications would
be ``{None: x}``, i.e., there is only one specification and it matches all argument
counts, then it is collapsed into just ``x``.

A specification is either a tuple or None. If a tuple, each element can be either a number
``n``, meaning that the nth argument should be extracted as a message, or the tuple
``(n, 'c')``, meaning that the nth argument should be extracted as context for the
messages. A ``None`` specification is equivalent to ``(1,)``, extracting the first
argument.

### Function: __getattr__(name)

### Function: __init__(self, dist)

### Function: initialize_options(self)

### Function: ensure_finalized(self)

### Function: finalize_options(self)

### Function: initialize_options(self)

### Function: finalize_options(self)

### Function: run(self)

### Function: _run_domain(self, domain)

### Function: cli_directory_filter(dirname)

### Function: initialize_options(self)

### Function: finalize_options(self)

### Function: _build_callback(self, path)

### Function: run(self)

### Function: _get_mappings(self)

### Function: initialize_options(self)

### Function: finalize_options(self)

### Function: run(self)

### Function: initialize_options(self)

### Function: finalize_options(self)

### Function: run(self)

### Function: run(self, argv)

**Description:** Main entry point of the command-line interface.

:param argv: list of arguments passed on the command-line

### Function: _configure_logging(self, loglevel)

### Function: _help(self)

### Function: _configure_command(self, cmdname, argv)

**Description:** :type cmdname: str
:type argv: list[str]

### Function: callback(filename, method, options)
