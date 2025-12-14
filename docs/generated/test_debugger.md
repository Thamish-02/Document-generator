## AI Summary

A file named test_debugger.py.


## Class: _FakeInput

**Description:** A fake input stream for pdb's interactive debugger.  Whenever a
line is read, print it (to simulate the user typing it), and then
return it.  The set of lines to return is specified in the
constructor; they should not have trailing newlines.

## Class: PdbTestInput

**Description:** Context manager that makes testing Pdb in doctests easier.

### Function: test_ipdb_magics()

**Description:** Test calling some IPython magics from ipdb.

First, set up some test functions and classes which we can inspect.

In [1]: class ExampleClass(object):
   ...:    """Docstring for ExampleClass."""
   ...:    def __init__(self):
   ...:        """Docstring for ExampleClass.__init__"""
   ...:        pass
   ...:    def __str__(self):
   ...:        return "ExampleClass()"

In [2]: def example_function(x, y, z="hello"):
   ...:     """Docstring for example_function."""
   ...:     pass

In [3]: old_trace = sys.gettrace()

Create a function which triggers ipdb.

In [4]: def trigger_ipdb():
   ...:    a = ExampleClass()
   ...:    debugger.Pdb().set_trace()

Run ipdb with faked input & check output. Because of a difference between
Python 3.13 & older versions, the first bit of the output is inconsistent.
We need to use ... to accommodate that, so the examples have to use IPython
prompts so that ... is distinct from the Python PS2 prompt.

In [5]: with PdbTestInput([
   ...:    'pdef example_function',
   ...:    'pdoc ExampleClass',
   ...:    'up',
   ...:    'down',
   ...:    'list',
   ...:    'pinfo a',
   ...:    'll',
   ...:    'continue',
   ...: ]):
   ...:     trigger_ipdb()
...> <doctest ...>(3)trigger_ipdb()
      1 def trigger_ipdb():
      2    a = ExampleClass()
----> 3    debugger.Pdb().set_trace()
<BLANKLINE>
ipdb> pdef example_function
 example_function(x, y, z='hello')
 ipdb> pdoc ExampleClass
Class docstring:
    Docstring for ExampleClass.
Init docstring:
    Docstring for ExampleClass.__init__
ipdb> up
> <doctest ...>(11)<module>()
      7    'pinfo a',
      8    'll',
      9    'continue',
     10 ]):
---> 11     trigger_ipdb()
<BLANKLINE>
ipdb> down...
> <doctest ...>(3)trigger_ipdb()
      1 def trigger_ipdb():
      2    a = ExampleClass()
----> 3    debugger.Pdb().set_trace()
<BLANKLINE>
ipdb> list
      1 def trigger_ipdb():
      2    a = ExampleClass()
----> 3    debugger.Pdb().set_trace()
<BLANKLINE>
ipdb> pinfo a
Type:           ExampleClass
String form:    ExampleClass()
Namespace:      Local...
Docstring:      Docstring for ExampleClass.
Init docstring: Docstring for ExampleClass.__init__
ipdb> ll
      1 def trigger_ipdb():
      2    a = ExampleClass()
----> 3    debugger.Pdb().set_trace()
<BLANKLINE>
ipdb> continue

Restore previous trace function, e.g. for coverage.py

In [6]: sys.settrace(old_trace)

### Function: test_ipdb_magics2()

**Description:** Test ipdb with a very short function.

>>> old_trace = sys.gettrace()

>>> def bar():
...     pass

Run ipdb.

>>> with PdbTestInput([
...    'continue',
... ]):
...     debugger.Pdb().runcall(bar)
> <doctest ...>(2)bar()
      1 def bar():
----> 2    pass
<BLANKLINE>
ipdb> continue

Restore previous trace function, e.g. for coverage.py    

>>> sys.settrace(old_trace)

### Function: can_quit()

**Description:** Test that quit work in ipydb

>>> old_trace = sys.gettrace()

>>> def bar():
...     pass

>>> with PdbTestInput([
...    'quit',
... ]):
...     debugger.Pdb().runcall(bar)
> <doctest ...>(2)bar()
        1 def bar():
----> 2    pass
<BLANKLINE>
ipdb> quit

Restore previous trace function, e.g. for coverage.py

>>> sys.settrace(old_trace)

### Function: can_exit()

**Description:** Test that quit work in ipydb

>>> old_trace = sys.gettrace()

>>> def bar():
...     pass

>>> with PdbTestInput([
...    'exit',
... ]):
...     debugger.Pdb().runcall(bar)
> <doctest ...>(2)bar()
        1 def bar():
----> 2    pass
<BLANKLINE>
ipdb> exit

Restore previous trace function, e.g. for coverage.py

>>> sys.settrace(old_trace)

### Function: test_interruptible_core_debugger()

**Description:** The debugger can be interrupted.

The presumption is there is some mechanism that causes a KeyboardInterrupt
(this is implemented in ipykernel).  We want to ensure the
KeyboardInterrupt cause debugging to cease.

### Function: test_xmode_skip()

**Description:** that xmode skip frames

Not as a doctest as pytest does not run doctests.

### Function: _decorator_skip_setup()

### Function: test_decorator_skip()

**Description:** test that decorator frames can be skipped.

### Function: test_decorator_skip_disabled()

**Description:** test that decorator frame skipping can be disabled

### Function: test_decorator_skip_with_breakpoint()

**Description:** test that decorator frame skipping can be disabled

### Function: test_where_erase_value()

**Description:** Test that `where` does not access f_locals and erase values.

### Function: __init__(self, lines)

### Function: readline(self)

### Function: __init__(self, input)

### Function: __enter__(self)

### Function: __exit__(self)

### Function: raising_input(msg, called)
