## AI Summary

A file named f2py2e.py.


### Function: scaninputline(inputline)

### Function: callcrackfortran(files, options)

### Function: buildmodules(lst)

### Function: dict_append(d_out, d_in)

### Function: run_main(comline_list)

**Description:** Equivalent to running::

    f2py <args>

where ``<args>=string.join(<list>,' ')``, but in Python.  Unless
``-h`` is used, this function returns a dictionary containing
information on generated modules and their dependencies on source
files.

You cannot build extension modules with this function, that is,
using ``-c`` is not allowed. Use the ``compile`` command instead.

Examples
--------
The command ``f2py -m scalar scalar.f`` can be executed from Python as
follows.

.. literalinclude:: ../../source/f2py/code/results/run_main_session.dat
    :language: python

### Function: filter_files(prefix, suffix, files, remove_prefix)

**Description:** Filter files by prefix and suffix.

### Function: get_prefix(module)

## Class: CombineIncludePaths

### Function: f2py_parser()

### Function: get_newer_options(iline)

### Function: make_f2py_compile_parser()

### Function: preparse_sysargv()

### Function: run_compile()

**Description:** Do it all in one call!

### Function: validate_modulename(pyf_files, modulename)

### Function: main()

### Function: __call__(self, parser, namespace, values, option_string)
