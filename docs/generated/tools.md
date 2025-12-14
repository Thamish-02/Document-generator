## AI Summary

A file named tools.py.


### Function: full_path(startPath, files)

**Description:** Make full paths for all the listed files, based on startPath.

Only the base part of startPath is kept, since this routine is typically
used with a script's ``__file__`` variable as startPath. The base of startPath
is then prepended to all the listed files, forming the output list.

Parameters
----------
startPath : string
  Initial path to use as the base for the results.  This path is split
  using os.path.split() and only its first component is kept.

files : list
  One or more files.

Examples
--------

>>> full_path('/foo/bar.py',['a.txt','b.txt'])
['/foo/a.txt', '/foo/b.txt']

>>> full_path('/foo',['a.txt','b.txt'])
['/a.txt', '/b.txt']

### Function: parse_test_output(txt)

**Description:** Parse the output of a test run and return errors, failures.

Parameters
----------
txt : str
  Text output of a test run, assumed to contain a line of one of the
  following forms::

    'FAILED (errors=1)'
    'FAILED (failures=1)'
    'FAILED (errors=1, failures=1)'

Returns
-------
nerr, nfail
  number of errors and failures.

### Function: default_argv()

**Description:** Return a valid default argv for creating testing instances of ipython

### Function: default_config()

**Description:** Return a config object with good defaults for testing.

### Function: get_ipython_cmd(as_string)

**Description:** Return appropriate IPython command line name. By default, this will return
a list that can be used with subprocess.Popen, for example, but passing
`as_string=True` allows for returning the IPython command as a string.

Parameters
----------
as_string: bool
    Flag to allow to return the command as a string.

### Function: ipexec(fname, options, commands)

**Description:** Utility to call 'ipython filename'.

Starts IPython with a minimal and safe configuration to make startup as fast
as possible.

Note that this starts IPython in a subprocess!

Parameters
----------
fname : str, Path
  Name of file to be executed (should have .py or .ipy extension).

options : optional, list
  Extra command-line flags to be passed to IPython.

commands : optional, list
  Commands to send in on stdin

Returns
-------
``(stdout, stderr)`` of ipython subprocess.

### Function: ipexec_validate(fname, expected_out, expected_err, options, commands)

**Description:** Utility to call 'ipython filename' and validate output/error.

This function raises an AssertionError if the validation fails.

Note that this starts IPython in a subprocess!

Parameters
----------
fname : str, Path
  Name of the file to be executed (should have .py or .ipy extension).

expected_out : str
  Expected stdout of the process.

expected_err : optional, str
  Expected stderr of the process.

options : optional, list
  Extra command-line flags to be passed to IPython.

Returns
-------
None

## Class: TempFileMixin

**Description:** Utility class to create temporary Python/IPython files.

Meant as a mixin class for test cases.

### Function: check_pairs(func, pairs)

**Description:** Utility function for the common case of checking a function with a
sequence of input/output pairs.

Parameters
----------
func : callable
  The function to be tested. Should accept a single argument.
pairs : iterable
  A list of (input, expected_output) tuples.

Returns
-------
None. Raises an AssertionError if any output does not match the expected
value.

## Class: AssertPrints

**Description:** Context manager for testing that code prints certain text.

Examples
--------
>>> with AssertPrints("abc", suppress=False):
...     print("abcd")
...     print("def")
...
abcd
def

## Class: AssertNotPrints

**Description:** Context manager for checking that certain output *isn't* produced.

Counterpart of AssertPrints

### Function: mute_warn()

### Function: make_tempfile(name)

**Description:** Create an empty, named, temporary file for the duration of the context.

### Function: fake_input(inputs)

**Description:** Temporarily replace the input() function to return the given values

Use as a context manager:

with fake_input(['result1', 'result2']):
    ...

Values are returned in order. If input() is called again after the last value
was used, EOFError is raised.

### Function: help_output_test(subcommand)

**Description:** test that `ipython [subcommand] -h` works

### Function: help_all_output_test(subcommand)

**Description:** test that `ipython [subcommand] --help-all` works

### Function: mktmp(self, src, ext)

**Description:** Make a valid python temp file.

### Function: tearDown(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, s, channel, suppress)

### Function: __enter__(self)

### Function: __exit__(self, etype, value, traceback)

### Function: __exit__(self, etype, value, traceback)

### Function: mock_input(prompt)
