## AI Summary

A file named test_magic.py.


## Class: DummyMagics

### Function: test_extract_code_ranges()

### Function: test_extract_symbols()

### Function: test_extract_symbols_raises_exception_with_non_python_code()

### Function: test_magic_not_found()

### Function: test_cell_magic_not_found()

### Function: test_magic_error_status()

### Function: test_config()

**Description:** test that config magic does not raise
can happen if Configurable init is moved too early into
Magics.__init__ as then a Config object will be registered as a
magic.

### Function: test_config_available_configs()

**Description:** test that config magic prints available configs in unique and
sorted order. 

### Function: test_config_print_class()

**Description:** test that config with a classname prints the class's options. 

### Function: test_rehashx()

### Function: test_magic_parse_options()

**Description:** Test that we don't mangle paths when parsing magic options.

### Function: test_magic_parse_long_options()

**Description:** Magic.parse_options can handle --foo=bar long options

### Function: doctest_hist_f()

**Description:** Test %hist -f with temporary filename.

In [9]: import tempfile

In [10]: tfile = tempfile.mktemp('.py','tmp-ipython-')

In [11]: %hist -nl -f $tfile 3

In [13]: import os; os.unlink(tfile)

### Function: doctest_hist_op()

**Description:** Test %hist -op

In [1]: class b(float):
   ...:     pass
   ...:

In [2]: class s(object):
   ...:     def __str__(self):
   ...:         return 's'
   ...:

In [3]:

In [4]: class r(b):
   ...:     def __repr__(self):
   ...:         return 'r'
   ...:

In [5]: class sr(s,r): pass
   ...:

In [6]:

In [7]: bb=b()

In [8]: ss=s()

In [9]: rr=r()

In [10]: ssrr=sr()

In [11]: 4.5
Out[11]: 4.5

In [12]: str(ss)
Out[12]: 's'

In [13]:

In [14]: %hist -op
>>> class b:
...     pass
...
>>> class s(b):
...     def __str__(self):
...         return 's'
...
>>>
>>> class r(b):
...     def __repr__(self):
...         return 'r'
...
>>> class sr(s,r): pass
>>>
>>> bb=b()
>>> ss=s()
>>> rr=r()
>>> ssrr=sr()
>>> 4.5
4.5
>>> str(ss)
's'
>>>

### Function: test_hist_pof()

### Function: test_macro()

### Function: test_macro_run()

**Description:** Test that we can run a multi-line macro successfully.

### Function: test_magic_magic()

**Description:** Test %magic

### Function: test_numpy_reset_array_undec()

**Description:** Test '%reset array' functionality

### Function: test_reset_out()

**Description:** Test '%reset out' magic

### Function: test_reset_in()

**Description:** Test '%reset in' magic

### Function: test_reset_dhist()

**Description:** Test '%reset dhist' magic

### Function: test_reset_in_length()

**Description:** Test that '%reset in' preserves In[] length

## Class: TestResetErrors

### Function: test_tb_syntaxerror()

**Description:** test %tb after a SyntaxError

### Function: test_time()

### Function: test_time_no_output_with_semicolon()

### Function: test_time_last_not_expression()

### Function: test_time2()

### Function: test_time3()

**Description:** Erroneous magic function calls, issue gh-3334

### Function: test_multiline_time()

**Description:** Make sure last statement from time return a value.

### Function: test_time_local_ns()

**Description:** Test that local_ns is actually global_ns when running a cell magic

### Function: test_time_microseconds_display()

**Description:** Ensure ASCII is used when necessary

### Function: test_capture()

### Function: test_doctest_mode()

**Description:** Toggle doctest_mode twice, it should be a no-op and run without error

### Function: test_parse_options()

**Description:** Tests for basic options parsing in magics.

### Function: test_parse_options_preserve_non_option_string()

**Description:** Test to assert preservation of non-option part of magic-block, while parsing magic options.

### Function: test_run_magic_preserve_code_block()

**Description:** Test to assert preservation of non-option part of magic-block, while running magic.

### Function: test_dirops()

**Description:** Test various directory handling operations.

### Function: test_cd_force_quiet()

**Description:** Test OSMagics.cd_force_quiet option

### Function: test_xmode()

### Function: test_reset_hard()

## Class: TestXdel

### Function: doctest_who()

**Description:** doctest for %who

In [1]: %reset -sf

In [2]: alpha = 123

In [3]: beta = 'beta'

In [4]: %who int
alpha

In [5]: %who str
beta

In [6]: %whos
Variable   Type    Data/Info
----------------------------
alpha      int     123
beta       str     beta

In [7]: %who_ls
Out[7]: ['alpha', 'beta']

### Function: test_whos()

**Description:** Check that whos is protected against objects where repr() fails.

### Function: doctest_precision()

**Description:** doctest for %precision

In [1]: f = get_ipython().display_formatter.formatters['text/plain']

In [2]: %precision 5
Out[2]: '%.5f'

In [3]: f.float_format
Out[3]: '%.5f'

In [4]: %precision %e
Out[4]: '%e'

In [5]: f(3.1415927)
Out[5]: '3.141593e+00'

### Function: test_debug_magic()

**Description:** Test debugging a small code with %debug

In [1]: with PdbTestInput(['c']):
   ...:     %debug print("a b") #doctest: +ELLIPSIS
   ...:
...
ipdb> c
a b
In [2]:

### Function: test_debug_magic_locals()

**Description:** Test debugging a small code with %debug with locals

In [1]: with PdbTestInput(['c']):
   ...:     def fun():
   ...:         res = 1
   ...:         %debug print(res)
   ...:     fun()
   ...:
...
ipdb> c
1
In [2]:

### Function: test_psearch()

### Function: test_timeit_shlex()

**Description:** test shlex issues with timeit (#1109)

### Function: test_timeit_special_syntax()

**Description:** Test %%timeit with IPython special syntax

### Function: test_timeit_return()

**Description:** test whether timeit -o return object

### Function: test_timeit_quiet()

**Description:** test quiet option of timeit magic

### Function: test_timeit_return_quiet()

### Function: test_timeit_invalid_return()

### Function: test_timeit_raise_on_interrupt()

### Function: test_prun_special_syntax()

**Description:** Test %%prun with IPython special syntax

### Function: test_prun_quotes()

**Description:** Test that prun does not clobber string escapes (GH #1302)

### Function: test_extension()

### Function: test_notebook_export_json()

## Class: TestEnv

## Class: CellMagicTestCase

### Function: test_file()

**Description:** Basic %%writefile

### Function: test_file_single_quote()

**Description:** Basic %%writefile with embedded single quotes

### Function: test_file_double_quote()

**Description:** Basic %%writefile with embedded double quotes

### Function: test_file_var_expand()

**Description:** %%writefile $filename

### Function: test_file_unicode()

**Description:** %%writefile with unicode cell

### Function: test_file_amend()

**Description:** %%writefile -a amends files

### Function: test_file_spaces()

**Description:** %%file with spaces in filename

### Function: test_script_config()

### Function: _interrupt_after_1s()

### Function: test_script_raise_on_interrupt()

### Function: test_script_do_not_raise_on_interrupt()

### Function: test_script_out()

### Function: test_script_err()

### Function: test_script_out_err()

### Function: test_script_defaults()

## Class: FooFoo

**Description:** class with both %foo and %%foo magics

### Function: test_line_cell_info()

**Description:** %%foo and %foo magics are distinguishable to inspect

### Function: test_multiple_magics()

### Function: test_alias_magic()

**Description:** Test %alias_magic.

### Function: test_save()

**Description:** Test %save.

### Function: test_save_with_no_args()

### Function: test_store()

**Description:** Test %store.

### Function: _run_edit_test(arg_s, exp_filename, exp_lineno, exp_contents, exp_is_temp)

### Function: test_edit_interactive()

**Description:** %edit on interactively defined objects

### Function: test_edit_cell()

**Description:** %edit [cell id]

### Function: test_edit_fname()

**Description:** %edit file

### Function: test_bookmark()

### Function: test_ls_magic()

### Function: test_strip_initial_indent()

### Function: test_logging_magic_quiet_from_arg()

### Function: test_logging_magic_quiet_from_config()

### Function: test_logging_magic_not_quiet()

### Function: test_time_no_var_expand()

### Function: test_timeit_arguments()

**Description:** Test valid timeit arguments, should not cause SyntaxError (GH #1269)

### Function: test_time_raise_on_interrupt()

### Function: test_lazy_magics()

### Function: test_run_module_from_import_hook()

**Description:** Test that a module can be loaded via an import hook

### Function: fail(shell)

### Function: test_reset_redefine(self)

## Class: A

### Function: test_xdel(self)

**Description:** Test that references from %run are cleared by xdel.

## Class: A

### Function: lmagic(line)

### Function: lmagic(line)

### Function: test_env(self)

### Function: test_env_secret(self)

### Function: test_env_get_set_simple(self)

### Function: test_env_get_set_complex(self)

### Function: test_env_set_bad_input(self)

### Function: test_env_set_whitespace(self)

### Function: check_ident(self, magic)

### Function: test_cell_magic_func_deco(self)

**Description:** Cell magic using simple decorator

### Function: test_cell_magic_reg(self)

**Description:** Cell magic manually registered

### Function: test_cell_magic_class(self)

**Description:** Cell magics declared via a class

### Function: test_cell_magic_class2(self)

**Description:** Cell magics declared via a class, #2

### Function: print_numbers()

### Function: line_foo(self, line)

**Description:** I am line foo

### Function: cell_foo(self, line, cell)

**Description:** I am cell foo, not line foo

### Function: sii(s)

## Class: KernelMagics

### Function: __del__(self)

### Function: __repr__(self)

### Function: __repr__(self)

### Function: cellm(line, cell)

### Function: cellm(line, cell)

## Class: MyMagics

## Class: MyMagics2

## Class: MyTempImporter

### Function: less(self, shell)

### Function: cellm3(self, line, cell)

### Function: cellm33(self, line, cell)

### Function: find_spec(self, fullname, path, target)

### Function: get_filename(self, fullname)

### Function: get_data(self, path)
