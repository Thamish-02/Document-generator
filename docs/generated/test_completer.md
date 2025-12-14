## AI Summary

A file named test_completer.py.


### Function: jedi_status(status)

### Function: recompute_unicode_ranges()

**Description:** utility to recompute the largest unicode range without any characters

use to recompute the gap in the global _UNICODE_RANGES of completer.py

### Function: test_unicode_range()

**Description:** Test that the ranges we test for unicode names give the same number of
results than testing the full length.

### Function: greedy_completion()

### Function: evaluation_policy(evaluation)

### Function: custom_matchers(matchers)

### Function: test_protect_filename(s1, expected)

### Function: check_line_split(splitter, test_specs)

### Function: test_line_split()

**Description:** Basic line splitter test with default specs.

## Class: NamedInstanceClass

## Class: KeyCompletable

## Class: TestCompleter

### Function: test_completion_context(line, expected)

**Description:** Test completion context

### Function: test_unsupported_completion_context(line, expected)

**Description:** Test unsupported completion context

### Function: test_misc_no_jedi_completions(setup, code, expected, not_expected)

### Function: test_trim_expr(code, expected)

### Function: test_match_numeric_literal_for_dict_key(input, expected)

### Function: ranges(i)

### Function: __init__(self, name)

### Function: _ipython_key_completions_(cls)

### Function: __init__(self, things)

### Function: _ipython_key_completions_(self)

### Function: setUp(self)

**Description:** We want to silence all PendingDeprecationWarning when testing the completer

### Function: tearDown(self)

### Function: test_custom_completion_error(self)

**Description:** Test that errors from custom attribute completers are silenced.

### Function: test_custom_completion_ordering(self)

**Description:** Test that errors from custom attribute completers are silenced.

### Function: test_unicode_completions(self)

### Function: test_latex_completions(self)

### Function: test_latex_no_results(self)

**Description:** forward latex should really return nothing in either field if nothing is found.

### Function: test_back_latex_completion(self)

### Function: test_back_unicode_completion(self)

### Function: test_forward_unicode_completion(self)

### Function: test_delim_setting(self)

### Function: test_spaces(self)

**Description:** Test with only spaces as split chars.

### Function: test_has_open_quotes1(self)

### Function: test_has_open_quotes2(self)

### Function: test_has_open_quotes3(self)

### Function: test_has_open_quotes4(self)

### Function: test_abspath_file_completions(self)

### Function: test_local_file_completions(self)

### Function: test_quoted_file_completions(self)

### Function: test_all_completions_dups(self)

**Description:** Make sure the output of `IPCompleter.all_completions` does not have
duplicated prefixes.

### Function: test_jedi(self)

**Description:** A couple of issue we had with Jedi

### Function: test_completion_have_signature(self)

**Description:** Lets make sure jedi is capable of pulling out the signature of the function we are completing.

### Function: test_completions_have_type(self)

**Description:** Lets make sure matchers provide completion type.

### Function: test_deduplicate_completions(self)

**Description:** Test that completions are correctly deduplicated (even if ranges are not the same)

### Function: test_greedy_completions(self)

**Description:** Test the capability of the Greedy completer.

Most of the test here does not really show off the greedy completer, for proof
each of the text below now pass with Jedi. The greedy completer is capable of more.

See the :any:`test_dict_key_completion_contexts`

### Function: test_omit__names(self)

### Function: test_limit_to__all__False_ok(self)

**Description:** Limit to all is deprecated, once we remove it this test can go away.

### Function: test_get__all__entries_ok(self)

### Function: test_get__all__entries_no__all__ok(self)

### Function: test_completes_globals_as_args_of_methods(self)

### Function: test_completes_attributes_in_fstring_expressions(self)

### Function: test_completes_in_dict_expressions(self)

### Function: test_func_kw_completions(self)

### Function: test_default_arguments_from_docstring(self)

### Function: test_line_magics(self)

### Function: test_cell_magics(self)

### Function: test_line_cell_magics(self)

### Function: test_magic_completion_order(self)

### Function: test_magic_completion_shadowing(self)

### Function: test_magic_completion_shadowing_explicit(self)

**Description:** If the user try to complete a shadowed magic, and explicit % start should
still return the completions.

### Function: test_magic_config(self)

### Function: test_magic_color(self)

### Function: test_match_dict_keys(self)

**Description:** Test that match_dict_keys works on a couple of use case does return what
expected, and does not crash

### Function: test_match_dict_keys_tuple(self)

**Description:** Test that match_dict_keys called with extra prefix works on a couple of use case,
does return what expected, and does not crash.

### Function: test_dict_key_completion_closures(self)

### Function: test_dict_key_completion_string(self)

**Description:** Test dictionary key completion for string keys

### Function: test_dict_key_completion_numbers(self)

### Function: test_dict_key_completion_contexts(self)

**Description:** Test expression contexts in which dict key completion occurs

### Function: test_dict_key_completion_bytes(self)

**Description:** Test handling of bytes in dict key completion

### Function: test_dict_key_completion_unicode_py3(self)

**Description:** Test handling of unicode in dict key completion

### Function: test_struct_array_key_completion(self)

**Description:** Test dict key completion applies to numpy struct arrays

### Function: test_dataframe_key_completion(self)

**Description:** Test dict key completion applies to pandas DataFrames

### Function: test_dict_key_completion_invalids(self)

**Description:** Smoke test cases dict key completion can't handle

### Function: test_object_key_completion(self)

### Function: test_class_key_completion(self)

### Function: test_tryimport(self)

**Description:** Test that try-import don't crash on trailing dot, and import modules before

### Function: test_aimport_module_completer(self)

### Function: test_nested_import_module_completer(self)

### Function: test_import_module_completer(self)

### Function: test_from_module_completer(self)

### Function: test_snake_case_completion(self)

### Function: test_mix_terms(self)

### Function: test_percent_symbol_restrict_to_magic_completions(self)

### Function: test_fwd_unicode_restricts(self)

### Function: test_dict_key_restrict_to_dicts(self)

**Description:** Test that dict key suppresses non-dict completion items

### Function: test_matcher_suppression(self)

### Function: test_matcher_suppression_with_iterator(self)

### Function: test_matcher_suppression_with_jedi(self)

### Function: test_matcher_disabling(self)

### Function: test_matcher_priority(self)

## Class: A

### Function: complete_A(a, existing_completions)

### Function: complete_example(a)

### Function: _(text)

### Function: _test_complete(reason, s, comp, start, end)

### Function: _test_not_complete(reason, s, comp)

### Function: _(line, cursor_pos, expect, message, completion)

## Class: A

## Class: A

## Class: CustomClass

### Function: _foo_cellm(line, cell)

### Function: _bar_cellm(line, cell)

### Function: match()

### Function: match()

## Class: C

### Function: assert_no_completion()

### Function: assert_completion()

### Function: assert_completion()

### Function: completes_on_nested()

### Function: _()

### Function: a_matcher(text)

### Function: b_matcher(context)

### Function: c_matcher(text)

### Function: matcher_returning_iterator(text)

### Function: matcher_returning_list(text)

### Function: configure(suppression_config)

### Function: _()

### Function: a_matcher(text)

### Function: b_matcher(text)

### Function: _(expected)

### Function: a_matcher(text)

### Function: b_matcher(text)

### Function: _(expected)

### Function: method_one(self)

### Function: _(text, expected)

### Function: configure(suppression_config)

### Function: _(text, expected)

### Function: configure(suppression_config)
