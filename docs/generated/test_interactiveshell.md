## AI Summary

A file named test_interactiveshell.py.


## Class: DerivedInterrupt

## Class: InteractiveShellTestCase

## Class: TestSafeExecfileNonAsciiPath

## Class: ExitCodeChecks

## Class: TestSystemRaw

### Function: test_magic_warnings(magic_cmd)

## Class: TestSystemPipedExitCode

## Class: TestModules

## Class: Negator

**Description:** Negates all number literals in an AST.

## Class: TestAstTransform

## Class: TestMiscTransform

## Class: IntegerWrapper

**Description:** Wraps all integers in a call to Integer()

## Class: TestAstTransform2

## Class: ErrorTransformer

**Description:** Throws an error when it sees a number.

## Class: TestAstTransformError

## Class: StringRejector

**Description:** Throws an InputRejected when it sees a string literal.

Used to verify that NodeTransformers can signal that a piece of code should
not be executed by throwing an InputRejected.

## Class: TestAstTransformInputRejection

### Function: test__IPYTHON__()

## Class: DummyRepr

### Function: test_user_variables()

### Function: test_user_expression()

## Class: TestSyntaxErrorTransformer

**Description:** Check that SyntaxError raised by an input transformer is handled by run_cell()

## Class: TestWarningSuppression

## Class: TestImportNoDeprecate

### Function: test_custom_exc_count()

### Function: test_run_cell_async()

### Function: test_run_cell_await()

### Function: test_run_cell_asyncio_run()

### Function: test_should_run_async()

### Function: test_set_custom_completer()

## Class: TestShowTracebackAttack

**Description:** Test that the interactive shell is resilient against the client attack of
manipulating the showtracebacks method. These attacks shouldn't result in an
unhandled exception in the kernel.

### Function: test_enable_gui_osx()

### Function: test_naked_string_cells(self)

**Description:** Test that cells with only naked strings are fully executed

### Function: test_run_empty_cell(self)

**Description:** Just make sure we don't get a horrible error with a blank
cell of input. Yes, I did overlook that.

### Function: test_run_cell_multiline(self)

**Description:** Multi-block, multi-line cells must execute correctly.
        

### Function: test_multiline_string_cells(self)

**Description:** Code sprinkled with multiline strings should execute (GH-306)

### Function: test_dont_cache_with_semicolon(self)

**Description:** Ending a line with semicolon should not cache the returned object (GH-307)

### Function: test_syntax_error(self)

### Function: test_open_standard_input_stream(self)

### Function: test_open_standard_output_stream(self)

### Function: test_open_standard_error_stream(self)

### Function: test_In_variable(self)

**Description:** Verify that In variable grows with user input (GH-284)

### Function: test_magic_names_in_string(self)

### Function: test_trailing_newline(self)

**Description:** test that running !(command) does not raise a SyntaxError

### Function: test_gh_597(self)

**Description:** Pretty-printing lists of objects with non-ascii reprs may cause
problems.

### Function: test_future_flags(self)

**Description:** Check that future flags are used for parsing code (gh-777)

### Function: test_can_pickle(self)

**Description:** Can we pickle objects defined interactively (GH-29)

### Function: test_global_ns(self)

**Description:** Code in functions must be able to access variables outside them.

### Function: test_bad_custom_tb(self)

**Description:** Check that InteractiveShell is protected from bad custom exception handlers

### Function: test_bad_custom_tb_return(self)

**Description:** Check that InteractiveShell is protected from bad return types in custom exception handlers

### Function: test_drop_by_id(self)

### Function: test_var_expand(self)

### Function: test_var_expand_local(self)

**Description:** Test local variable expansion in !system and %magic calls

### Function: test_var_expand_self(self)

**Description:** Test variable expansion with the name 'self', which was failing.

See https://github.com/ipython/ipython/issues/1878#issuecomment-7698218

### Function: test_bad_var_expand(self)

**Description:** var_expand on invalid formats shouldn't raise

### Function: test_silent_postexec(self)

**Description:** run_cell(silent=True) doesn't invoke pre/post_run_cell callbacks

### Function: test_silent_noadvance(self)

**Description:** run_cell(silent=True) doesn't advance execution_count

### Function: test_silent_nodisplayhook(self)

**Description:** run_cell(silent=True) doesn't trigger displayhook

### Function: test_ofind_line_magic(self)

### Function: test_ofind_cell_magic(self)

### Function: test_ofind_property_with_error(self)

### Function: test_ofind_multiple_attribute_lookups(self)

### Function: test_ofind_slotted_attributes(self)

### Function: test_ofind_prefers_property_to_instance_level_attribute(self)

### Function: test_custom_syntaxerror_exception(self)

### Function: test_custom_exception(self)

### Function: test_showtraceback_with_surrogates(self, mocked_print)

### Function: test_mktempfile(self)

### Function: test_new_main_mod(self)

### Function: test_get_exception_only(self)

### Function: test_inspect_text(self)

### Function: test_last_execution_result(self)

**Description:** Check that last execution result gets set correctly (GH-10702) 

### Function: test_reset_aliasing(self)

**Description:** Check that standard posix aliases work after %reset. 

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_1(self)

**Description:** Test safe_execfile with non-ascii path
        

### Function: setUp(self)

### Function: test_exit_code_ok(self)

### Function: test_exit_code_error(self)

### Function: test_exit_code_signal(self)

### Function: test_exit_code_signal_csh(self)

### Function: setUp(self)

### Function: test_1(self)

**Description:** Test system_raw with non-ascii cmd
        

### Function: test_control_c(self)

### Function: setUp(self)

### Function: test_exit_code_ok(self)

### Function: test_exit_code_error(self)

### Function: test_exit_code_signal(self)

### Function: test_extraneous_loads(self)

**Description:** Test we're not loading modules on startup that we shouldn't.
        

### Function: visit_Num(self, node)

### Function: visit_Constant(self, node)

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_non_int_const(self)

### Function: test_run_cell(self)

### Function: test_timeit(self)

### Function: test_time(self)

### Function: test_macro(self)

### Function: test_transform_only_once(self)

### Function: visit_Num(self, node)

### Function: visit_Constant(self, node)

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_run_cell(self)

### Function: test_run_cell_non_int(self)

### Function: test_timeit(self)

### Function: visit_Constant(self, node)

### Function: test_unregistering(self)

### Function: visit_Constant(self, node)

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_input_rejection(self)

**Description:** Check that NodeTransformers can reject input.

### Function: __repr__(self)

### Function: _repr_html_(self)

### Function: _repr_javascript_(self)

### Function: transformer(lines)

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_syntaxerror_input_transformer(self)

### Function: test_warning_suppression(self)

### Function: test_deprecation_warning(self)

### Function: setUp(self)

**Description:** Make a valid python temp file.

### Function: test_no_dep(self)

**Description:** No deprecation warning should be raised from imported functions

### Function: foo()

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_set_show_tracebacks_none(self)

**Description:** Test the case of the client setting showtracebacks to None

### Function: test_set_show_tracebacks_noop(self)

**Description:** Test the case of the client setting showtracebacks to a no op lambda

## Class: Spam

### Function: failing_hook()

### Function: lmagic(line)

**Description:** A line magic

### Function: cmagic(line, cell)

**Description:** A cell magic

## Class: A

## Class: A

## Class: A

## Class: A

### Function: my_handler(shell, etype, value, tb, tb_offset)

### Function: my_handler(shell, etype, value, tb, tb_offset)

### Function: mock_print_func(value, sep, end, file, flush)

### Function: f(x)

### Function: f(x)

### Function: count_cleanup(lines)

### Function: count_line_t(lines)

### Function: Integer()

### Function: f(x)

### Function: __repr__(self)

### Function: foo(self)

### Function: foo(self)

### Function: __init__(self)

### Function: foo(self)
