## AI Summary

A file named test_crackfortran.py.


## Class: TestNoSpace

## Class: TestPublicPrivate

## Class: TestModuleProcedure

## Class: TestExternal

## Class: TestCrackFortran

## Class: TestMarkinnerspaces

## Class: TestDimSpec

**Description:** This test suite tests various expressions that are used as dimension
specifications.

There exists two usage cases where analyzing dimensions
specifications are important.

In the first case, the size of output arrays must be defined based
on the inputs to a Fortran function. Because Fortran supports
arbitrary bases for indexing, for instance, `arr(lower:upper)`,
f2py has to evaluate an expression `upper - lower + 1` where
`lower` and `upper` are arbitrary expressions of input parameters.
The evaluation is performed in C, so f2py has to translate Fortran
expressions to valid C expressions (an alternative approach is
that a developer specifies the corresponding C expressions in a
.pyf file).

In the second case, when user provides an input array with a given
size but some hidden parameters used in dimensions specifications
need to be determined based on the input array size. This is a
harder problem because f2py has to solve the inverse problem: find
a parameter `p` such that `upper(p) - lower(p) + 1` equals to the
size of input array. In the case when this equation cannot be
solved (e.g. because the input array size is wrong), raise an
error before calling the Fortran function (that otherwise would
likely crash Python process when the size of input arrays is
wrong). f2py currently supports this case only when the equation
is linear with respect to unknown parameter.

## Class: TestModuleDeclaration

## Class: TestEval

## Class: TestFortranReader

## Class: TestUnicodeComment

## Class: TestNameArgsPatternBacktracking

## Class: TestFunctionReturn

## Class: TestFortranGroupCounters

## Class: TestF77CommonBlockReader

## Class: TestParamEval

## Class: TestLowerF2PYDirective

### Function: test_module(self)

### Function: test_defaultPrivate(self)

### Function: test_defaultPublic(self, tmp_path)

### Function: test_access_type(self, tmp_path)

### Function: test_nowrap_private_proceedures(self, tmp_path)

### Function: test_moduleOperators(self, tmp_path)

### Function: test_notPublicPrivate(self, tmp_path)

### Function: test_external_as_statement(self)

### Function: test_external_as_attribute(self)

### Function: test_gh2848(self)

### Function: test_common_with_division(self)

### Function: test_do_not_touch_normal_spaces(self)

### Function: test_one_relevant_space(self)

### Function: test_ignore_inner_quotes(self)

### Function: test_multiple_relevant_spaces(self)

### Function: test_array_size(self, dimspec)

### Function: test_inv_array_size(self, dimspec)

### Function: test_dependencies(self, tmp_path)

### Function: test_eval_scalar(self)

### Function: test_input_encoding(self, tmp_path, encoding)

### Function: test_encoding_comment(self)

### Function: test_nameargspattern_backtracking(self, adversary)

**Description:** address ReDOS vulnerability:
https://github.com/numpy/numpy/issues/23338

### Function: test_function_rettype(self)

### Function: test_end_if_comment(self)

### Function: test_gh22648(self, tmp_path)

### Function: test_param_eval_nested(self)

### Function: test_param_eval_nonstandard_range(self)

### Function: test_param_eval_empty_range(self)

### Function: test_param_eval_non_array_param(self)

### Function: test_param_eval_too_many_dims(self)

### Function: test_no_lower_fail(self)

### Function: incr(x)

### Function: incr(x)
