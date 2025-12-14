## AI Summary

A file named test_run.py.


### Function: doctest_refbug()

**Description:** Very nasty problem with references held by multiple runs of a script.
See: https://github.com/ipython/ipython/issues/141

In [1]: _ip.clear_main_mod_cache()
# random

In [2]: %run refbug

In [3]: call_f()
lowercased: hello

In [4]: %run refbug

In [5]: call_f()
lowercased: hello
lowercased: hello

### Function: doctest_run_builtins()

**Description:** Check that %run doesn't damage __builtins__.

In [1]: import tempfile

In [2]: bid1 = id(__builtins__)

In [3]: fname = tempfile.mkstemp('.py')[1]

In [3]: f = open(fname, 'w', encoding='utf-8')

In [4]: dummy= f.write('pass\n')

In [5]: f.flush()

In [6]: t1 = type(__builtins__)

In [7]: %run $fname

In [7]: f.close()

In [8]: bid2 = id(__builtins__)

In [9]: t2 = type(__builtins__)

In [10]: t1 == t2
Out[10]: True

In [10]: bid1 == bid2
Out[10]: True

In [12]: try:
   ....:     os.unlink(fname)
   ....: except:
   ....:     pass
   ....:

### Function: doctest_run_option_parser()

**Description:** Test option parser in %run.

In [1]: %run print_argv.py
[]

In [2]: %run print_argv.py print*.py
['print_argv.py']

In [3]: %run -G print_argv.py print*.py
['print*.py']

### Function: doctest_run_option_parser_for_posix()

**Description:** Test option parser in %run (Linux/OSX specific).

You need double quote to escape glob in POSIX systems:

In [1]: %run print_argv.py print\\*.py
['print*.py']

You can't use quote to escape glob in POSIX systems:

In [2]: %run print_argv.py 'print*.py'
['print_argv.py']

### Function: doctest_run_option_parser_for_windows()

**Description:** Test option parser in %run (Windows specific).

In Windows, you can't escape ``*` `by backslash:

In [1]: %run print_argv.py print\\*.py
['print\\\\*.py']

You can use quote to escape glob:

In [2]: %run print_argv.py 'print*.py'
["'print*.py'"]

### Function: doctest_reset_del()

**Description:** Test that resetting doesn't cause errors in __del__ methods.

In [2]: class A(object):
   ...:     def __del__(self):
   ...:         print(str("Hi"))
   ...:

In [3]: a = A()

In [4]: get_ipython().reset(); import gc; x = gc.collect(0)
Hi

In [5]: 1+1
Out[5]: 2

## Class: TestMagicRunPass

## Class: TestMagicRunSimple

## Class: TestMagicRunWithPackage

### Function: test_run__name__()

### Function: test_run_tb()

**Description:** Test traceback offset in %run

### Function: test_multiprocessing_run()

**Description:** Set we can run mutiprocesgin without messing up up main namespace

Note that import `nose.tools as nt` modify the values
sys.module['__mp_main__'] so we need to temporarily set it to None to test
the issue.

### Function: test_script_tb()

**Description:** Test traceback offset in `ipython script.py`

### Function: setUp(self)

### Function: run_tmpfile(self)

### Function: run_tmpfile_p(self)

### Function: test_builtins_id(self)

**Description:** Check that %run doesn't damage __builtins__ 

### Function: test_builtins_type(self)

**Description:** Check that the type of __builtins__ doesn't change with %run.

However, the above could pass if __builtins__ was already modified to
be a dict (it should be a module) by a previous use of %run.  So we
also check explicitly that it really is a module:

### Function: test_run_profile(self)

**Description:** Test that the option -p, which invokes the profiler, do not
crash by invoking execfile

### Function: test_run_debug_twice(self)

### Function: test_run_debug_twice_with_breakpoint(self)

**Description:** Make a valid python temp file.

### Function: test_simpledef(self)

**Description:** Test that simple class definitions work.

### Function: test_obj_del(self)

**Description:** Test that object's __del__ methods are called on exit.

### Function: test_aggressive_namespace_cleanup(self)

**Description:** Test that namespace cleanup is not too aggressive GH-238

Returning from another run magic deletes the namespace

### Function: test_run_second(self)

**Description:** Test that running a second file doesn't clobber the first, gh-3547

### Function: test_tclass(self)

### Function: test_run_i_after_reset(self)

**Description:** Check that %run -i still works after %reset (gh-693)

### Function: test_unicode(self)

**Description:** Check that files in odd encodings are accepted.

### Function: test_run_py_file_attribute(self)

**Description:** Test handling of `__file__` attribute in `%run <file>.py`.

### Function: test_run_ipy_file_attribute(self)

**Description:** Test handling of `__file__` attribute in `%run <file.ipy>`.

### Function: test_run_formatting(self)

**Description:** Test that %run -t -N<N> does not raise a TypeError for N > 1.

### Function: test_ignore_sys_exit(self)

**Description:** Test the -e option to ignore sys.exit()

### Function: test_run_nb(self)

**Description:** Test %run notebook.ipynb

### Function: test_run_nb_error(self)

**Description:** Test %run notebook.ipynb error

### Function: test_file_options(self)

### Function: writefile(self, name, content)

### Function: setUp(self)

### Function: tearDown(self)

### Function: check_run_submodule(self, submodule, opts)

### Function: test_run_submodule_with_absolute_import(self)

### Function: test_run_submodule_with_relative_import(self)

**Description:** Run submodule that has a relative import statement (#2727).

### Function: test_prun_submodule_with_absolute_import(self)

### Function: test_prun_submodule_with_relative_import(self)

### Function: with_fake_debugger(func)

### Function: test_debug_run_submodule_with_absolute_import(self)

### Function: test_debug_run_submodule_with_relative_import(self)

### Function: test_module_options(self)

### Function: test_module_options_with_separator(self)

### Function: wrapper()
