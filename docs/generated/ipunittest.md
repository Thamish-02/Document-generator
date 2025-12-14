## AI Summary

A file named ipunittest.py.


### Function: count_failures(runner)

**Description:** Count number of failures in a doctest runner.

Code modeled after the summarize() method in doctest.

## Class: IPython2PythonConverter

**Description:** Convert IPython 'syntax' to valid Python.

Eventually this code may grow to be the full IPython syntax conversion
implementation, but for now it only does prompt conversion.

## Class: Doc2UnitTester

**Description:** Class whose instances act as a decorator for docstring testing.

In practice we're only likely to need one instance ever, made below (though
no attempt is made at turning it into a singleton, there is no need for
that).

### Function: ipdocstring(func)

**Description:** Change the function docstring via ip2py.
    

### Function: __init__(self)

### Function: __call__(self, ds)

**Description:** Convert IPython prompts to python ones in a string.

### Function: __init__(self, verbose)

**Description:** New decorator.

Parameters
----------

verbose : boolean, optional (False)
  Passed to the doctest finder and runner to control verbosity.

### Function: __call__(self, func)

**Description:** Use as a decorator: doctest a function's docstring as a unittest.

This version runs normal doctests, but the idea is to make it later run
ipython syntax instead.

## Class: Tester

### Function: test(self)
