## AI Summary

A file named test_exceptions.py.


## Class: TestBestMatch

## Class: TestByRelevance

## Class: TestErrorTree

## Class: TestErrorInitReprStr

## Class: TestHashable

## Class: TestJsonPathRendering

### Function: best_match_of(self, instance, schema)

### Function: test_shallower_errors_are_better_matches(self)

### Function: test_oneOf_and_anyOf_are_weak_matches(self)

**Description:** A property you *must* match is probably better than one you have to
match a part of.

### Function: test_if_the_most_relevant_error_is_anyOf_it_is_traversed(self)

**Description:** If the most relevant error is an anyOf, then we traverse its context
and select the otherwise *least* relevant error, since in this case
that means the most specific, deep, error inside the instance.

I.e. since only one of the schemas must match, we look for the most
relevant one.

### Function: test_no_anyOf_traversal_for_equally_relevant_errors(self)

**Description:** We don't traverse into an anyOf (as above) if all of its context errors
seem to be equally "wrong" against the instance.

### Function: test_anyOf_traversal_for_single_equally_relevant_error(self)

**Description:** We *do* traverse anyOf with a single nested error, even though it is
vacuously equally relevant to itself.

### Function: test_anyOf_traversal_for_single_sibling_errors(self)

**Description:** We *do* traverse anyOf with a single subschema that fails multiple
times (e.g. on multiple items).

### Function: test_anyOf_traversal_for_non_type_matching_sibling_errors(self)

**Description:** We *do* traverse anyOf with multiple subschemas when one does not type
match.

### Function: test_if_the_most_relevant_error_is_oneOf_it_is_traversed(self)

**Description:** If the most relevant error is an oneOf, then we traverse its context
and select the otherwise *least* relevant error, since in this case
that means the most specific, deep, error inside the instance.

I.e. since only one of the schemas must match, we look for the most
relevant one.

### Function: test_no_oneOf_traversal_for_equally_relevant_errors(self)

**Description:** We don't traverse into an oneOf (as above) if all of its context errors
seem to be equally "wrong" against the instance.

### Function: test_oneOf_traversal_for_single_equally_relevant_error(self)

**Description:** We *do* traverse oneOf with a single nested error, even though it is
vacuously equally relevant to itself.

### Function: test_oneOf_traversal_for_single_sibling_errors(self)

**Description:** We *do* traverse oneOf with a single subschema that fails multiple
times (e.g. on multiple items).

### Function: test_oneOf_traversal_for_non_type_matching_sibling_errors(self)

**Description:** We *do* traverse oneOf with multiple subschemas when one does not type
match.

### Function: test_if_the_most_relevant_error_is_allOf_it_is_traversed(self)

**Description:** Now, if the error is allOf, we traverse but select the *most* relevant
error from the context, because all schemas here must match anyways.

### Function: test_nested_context_for_oneOf(self)

**Description:** We traverse into nested contexts (a oneOf containing an error in a
nested oneOf here).

### Function: test_it_prioritizes_matching_types(self)

### Function: test_it_prioritizes_matching_union_types(self)

### Function: test_boolean_schemas(self)

### Function: test_one_error(self)

### Function: test_no_errors(self)

### Function: test_short_paths_are_better_matches(self)

### Function: test_global_errors_are_even_better_matches(self)

### Function: test_weak_keywords_are_lower_priority(self)

### Function: test_strong_keywords_are_higher_priority(self)

### Function: test_it_knows_how_many_total_errors_it_contains(self)

### Function: test_it_contains_an_item_if_the_item_had_an_error(self)

### Function: test_it_does_not_contain_an_item_if_the_item_had_no_error(self)

### Function: test_keywords_that_failed_appear_in_errors_dict(self)

### Function: test_it_creates_a_child_tree_for_each_nested_path(self)

### Function: test_children_have_their_errors_dicts_built(self)

### Function: test_multiple_errors_with_instance(self)

### Function: test_it_does_not_contain_subtrees_that_are_not_in_the_instance(self)

### Function: test_if_its_in_the_tree_anyhow_it_does_not_raise_an_error(self)

**Description:** If a keyword refers to a path that isn't in the instance, the
tree still properly returns a subtree for that path.

### Function: test_iter(self)

### Function: test_repr_single(self)

### Function: test_repr_multiple(self)

### Function: test_repr_empty(self)

### Function: make_error(self)

### Function: assertShows(self, expected)

### Function: test_it_calls_super_and_sets_args(self)

### Function: test_repr(self)

### Function: test_unset_error(self)

### Function: test_empty_paths(self)

### Function: test_one_item_paths(self)

### Function: test_multiple_item_paths(self)

### Function: test_uses_pprint(self)

### Function: test_does_not_reorder_dicts(self)

### Function: test_str_works_with_instances_having_overriden_eq_operator(self)

**Description:** Check for #164 which rendered exceptions unusable when a
`ValidationError` involved instances with an `__eq__` method
that returned truthy values.

### Function: test_hashable(self)

### Function: validate_json_path_rendering(self, property_name, expected_path)

### Function: test_basic(self)

### Function: test_empty(self)

### Function: test_number(self)

### Function: test_period(self)

### Function: test_single_quote(self)

### Function: test_space(self)

### Function: test_backslash(self)

### Function: test_backslash_single_quote(self)

### Function: test_underscore(self)

### Function: test_double_quote(self)

### Function: test_hyphen(self)

### Function: test_json_path_injection(self)

### Function: test_open_bracket(self)

## Class: DontEQMeBro

### Function: __eq__(this, other)

### Function: __ne__(this, other)
