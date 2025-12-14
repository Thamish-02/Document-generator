## AI Summary

A file named test_validators.py.


### Function: fail(validator, errors, instance, schema)

## Class: TestCreateAndExtend

## Class: TestValidationErrorMessages

## Class: TestValidationErrorDetails

## Class: MetaSchemaTestsMixin

## Class: ValidatorTestMixin

## Class: AntiDraft6LeakMixin

**Description:** Make sure functionality from draft 6 doesn't leak backwards in time.

## Class: TestDraft3Validator

## Class: TestDraft4Validator

## Class: TestDraft6Validator

## Class: TestDraft7Validator

## Class: TestDraft201909Validator

## Class: TestDraft202012Validator

## Class: TestLatestValidator

**Description:** These really apply to multiple versions but are easiest to test on one.

## Class: TestValidatorFor

## Class: TestValidate

## Class: TestThreading

**Description:** Threading-related functionality tests.

jsonschema doesn't promise thread safety, and its validation behavior
across multiple threads may change at any time, but that means it isn't
safe to share *validators* across threads, not that anytime one has
multiple threads that jsonschema won't work (it certainly is intended to).

These tests ensure that this minimal level of functionality continues to
work.

## Class: TestReferencing

## Class: TestRefResolver

### Function: sorted_errors(errors)

## Class: ReallyFakeRequests

## Class: _ReallyFakeJSONResponse

### Function: setUp(self)

### Function: test_attrs(self)

### Function: test_init(self)

### Function: test_iter_errors_successful(self)

### Function: test_iter_errors_one_error(self)

### Function: test_iter_errors_multiple_errors(self)

### Function: test_if_a_version_is_provided_it_is_registered(self)

### Function: test_repr(self)

### Function: test_long_repr(self)

### Function: test_repr_no_version(self)

### Function: test_dashes_are_stripped_from_validator_names(self)

### Function: test_if_a_version_is_not_provided_it_is_not_registered(self)

### Function: test_validates_registers_meta_schema_id(self)

### Function: test_validates_registers_meta_schema_draft6_id(self)

### Function: test_create_default_types(self)

### Function: test_check_schema_with_different_metaschema(self)

**Description:** One can create a validator class whose metaschema uses a different
dialect than itself.

### Function: test_check_schema_with_different_metaschema_defaults_to_self(self)

**Description:** A validator whose metaschema doesn't declare $schema defaults to its
own validation behavior, not the latest "normal" specification.

### Function: test_extend(self)

### Function: test_extend_idof(self)

**Description:** Extending a validator preserves its notion of schema IDs.

### Function: test_extend_applicable_validators(self)

**Description:** Extending a validator preserves its notion of applicable validators.

### Function: message_for(self, instance, schema)

### Function: test_single_type_failure(self)

### Function: test_single_type_list_failure(self)

### Function: test_multiple_type_failure(self)

### Function: test_object_with_named_type_failure(self)

### Function: test_minimum(self)

### Function: test_maximum(self)

### Function: test_dependencies_single_element(self)

### Function: test_object_without_title_type_failure_draft3(self)

### Function: test_dependencies_list_draft3(self)

### Function: test_dependencies_list_draft7(self)

### Function: test_additionalItems_single_failure(self)

### Function: test_additionalItems_multiple_failures(self)

### Function: test_additionalProperties_single_failure(self)

### Function: test_additionalProperties_multiple_failures(self)

### Function: test_const(self)

### Function: test_contains_draft_6(self)

### Function: test_invalid_format_default_message(self)

### Function: test_additionalProperties_false_patternProperties(self)

### Function: test_False_schema(self)

### Function: test_multipleOf(self)

### Function: test_minItems(self)

### Function: test_maxItems(self)

### Function: test_minItems_1(self)

### Function: test_maxItems_0(self)

### Function: test_minLength(self)

### Function: test_maxLength(self)

### Function: test_minLength_1(self)

### Function: test_maxLength_0(self)

### Function: test_minProperties(self)

### Function: test_maxProperties(self)

### Function: test_minProperties_1(self)

### Function: test_maxProperties_0(self)

### Function: test_prefixItems_with_items(self)

### Function: test_prefixItems_with_multiple_extra_items(self)

### Function: test_pattern(self)

### Function: test_does_not_contain(self)

### Function: test_contains_too_few(self)

### Function: test_contains_too_few_both_constrained(self)

### Function: test_contains_too_many(self)

### Function: test_contains_too_many_both_constrained(self)

### Function: test_exclusiveMinimum(self)

### Function: test_exclusiveMaximum(self)

### Function: test_required(self)

### Function: test_dependentRequired(self)

### Function: test_oneOf_matches_none(self)

### Function: test_oneOf_matches_too_many(self)

### Function: test_unevaluated_items(self)

### Function: test_unevaluated_items_on_invalid_type(self)

### Function: test_unevaluated_properties_invalid_against_subschema(self)

### Function: test_unevaluated_properties_disallowed(self)

### Function: test_unevaluated_properties_on_invalid_type(self)

### Function: test_single_item(self)

### Function: test_heterogeneous_additionalItems_with_Items(self)

### Function: test_heterogeneous_items_prefixItems(self)

### Function: test_heterogeneous_unevaluatedItems_prefixItems(self)

### Function: test_heterogeneous_properties_additionalProperties(self)

**Description:** Not valid deserialized JSON, but this should not blow up.

### Function: test_heterogeneous_properties_unevaluatedProperties(self)

**Description:** Not valid deserialized JSON, but this should not blow up.

### Function: test_anyOf(self)

### Function: test_type(self)

### Function: test_single_nesting(self)

### Function: test_multiple_nesting(self)

### Function: test_recursive(self)

### Function: test_additionalProperties(self)

### Function: test_patternProperties(self)

### Function: test_additionalItems(self)

### Function: test_additionalItems_with_items(self)

### Function: test_propertyNames(self)

### Function: test_if_then(self)

### Function: test_if_else(self)

### Function: test_boolean_schema_False(self)

### Function: test_ref(self)

### Function: test_prefixItems(self)

### Function: test_prefixItems_with_items(self)

### Function: test_contains_too_many(self)

**Description:** `contains` + `maxContains` produces only one error, even if there are
many more incorrectly matching elements.

### Function: test_contains_too_few(self)

### Function: test_contains_none(self)

### Function: test_ref_sibling(self)

### Function: test_invalid_properties(self)

### Function: test_minItems_invalid_string(self)

### Function: test_enum_allows_empty_arrays(self)

**Description:** Technically, all the spec says is they SHOULD have elements, not MUST.

(As of Draft 6. Previous drafts do say MUST).

See #529.

### Function: test_enum_allows_non_unique_items(self)

**Description:** Technically, all the spec says is they SHOULD be unique, not MUST.

(As of Draft 6. Previous drafts do say MUST).

See #529.

### Function: test_schema_with_invalid_regex(self)

### Function: test_schema_with_invalid_regex_with_disabled_format_validation(self)

### Function: test_it_implements_the_validator_protocol(self)

### Function: test_valid_instances_are_valid(self)

### Function: test_invalid_instances_are_not_valid(self)

### Function: test_non_existent_properties_are_ignored(self)

### Function: test_evolve(self)

### Function: test_evolve_with_subclass(self)

**Description:** Subclassing validators isn't supported public API, but some users have
done it, because we don't actually error entirely when it's done :/

We need to deprecate doing so first to help as many of these users
ensure they can move to supported APIs, but this test ensures that in
the interim, we haven't broken those users.

### Function: test_is_type_is_true_for_valid_type(self)

### Function: test_is_type_is_false_for_invalid_type(self)

### Function: test_is_type_evades_bool_inheriting_from_int(self)

### Function: test_it_can_validate_with_decimals(self)

### Function: test_it_returns_true_for_formats_it_does_not_know_about(self)

### Function: test_it_does_not_validate_formats_by_default(self)

### Function: test_it_validates_formats_if_a_checker_is_provided(self)

### Function: test_non_string_custom_type(self)

### Function: test_it_properly_formats_tuples_in_errors(self)

**Description:** A tuple instance properly formats validation errors for uniqueItems.

See #224

### Function: test_check_redefined_sequence(self)

**Description:** Allow array to validate against another defined sequence type

### Function: test_it_creates_a_ref_resolver_if_not_provided(self)

### Function: test_it_upconverts_from_deprecated_RefResolvers(self)

### Function: test_it_upconverts_from_yet_older_deprecated_legacy_RefResolvers(self)

**Description:** Legacy RefResolvers support only the context manager form of
resolution.

### Function: test_True_is_not_a_schema(self)

### Function: test_False_is_not_a_schema(self)

### Function: test_True_is_not_a_schema_even_if_you_forget_to_check(self)

### Function: test_False_is_not_a_schema_even_if_you_forget_to_check(self)

### Function: test_any_type_is_valid_for_type_any(self)

### Function: test_any_type_is_redefinable(self)

**Description:** Sigh, because why not.

### Function: test_is_type_is_true_for_any_type(self)

### Function: test_is_type_does_not_evade_bool_if_it_is_being_tested(self)

### Function: test_ref_resolvers_may_have_boolean_schemas_stored(self)

### Function: test_draft_3(self)

### Function: test_draft_4(self)

### Function: test_draft_6(self)

### Function: test_draft_7(self)

### Function: test_draft_201909(self)

### Function: test_draft_202012(self)

### Function: test_True(self)

### Function: test_False(self)

### Function: test_custom_validator(self)

### Function: test_custom_validator_draft6(self)

### Function: test_validator_for_jsonschema_default(self)

### Function: test_validator_for_custom_default(self)

### Function: test_warns_if_meta_schema_specified_was_not_found(self)

### Function: test_does_not_warn_if_meta_schema_is_unspecified(self)

### Function: test_validator_for_custom_default_with_schema(self)

### Function: assertUses(self, schema, Validator)

### Function: test_draft3_validator_is_chosen(self)

### Function: test_draft4_validator_is_chosen(self)

### Function: test_draft6_validator_is_chosen(self)

### Function: test_draft7_validator_is_chosen(self)

### Function: test_draft202012_validator_is_chosen(self)

### Function: test_draft202012_validator_is_the_default(self)

### Function: test_validation_error_message(self)

### Function: test_schema_error_message(self)

### Function: test_it_uses_best_match(self)

### Function: test_validation_across_a_second_thread(self)

### Function: test_registry_with_retrieve(self)

### Function: test_custom_registries_do_not_autoretrieve_remote_resources(self)

### Function: setUp(self)

### Function: test_it_does_not_retrieve_schema_urls_from_the_network(self)

### Function: test_it_resolves_local_refs(self)

### Function: test_it_resolves_local_refs_with_id(self)

### Function: test_it_retrieves_stored_refs(self)

### Function: test_it_retrieves_unstored_refs_via_requests(self)

### Function: test_it_retrieves_unstored_refs_via_urlopen(self)

### Function: test_it_retrieves_local_refs_via_urlopen(self)

### Function: test_it_can_construct_a_base_uri_from_a_schema(self)

### Function: test_it_can_construct_a_base_uri_from_a_schema_without_id(self)

### Function: test_custom_uri_scheme_handlers(self)

### Function: test_cache_remote_on(self)

### Function: test_cache_remote_off(self)

### Function: test_if_you_give_it_junk_you_get_a_resolution_error(self)

### Function: test_helpful_error_message_on_failed_pop_scope(self)

### Function: test_pointer_within_schema_with_different_id(self)

**Description:** See #1085.

### Function: test_newly_created_validator_with_ref_resolver(self)

**Description:** See https://github.com/python-jsonschema/jsonschema/issues/1061#issuecomment-1624266555.

### Function: test_refresolver_with_pointer_in_schema_with_no_id(self)

**Description:** See https://github.com/python-jsonschema/jsonschema/issues/1124#issuecomment-1632574249.

### Function: key(error)

### Function: get(self, url)

### Function: json(self)

### Function: id_of(schema)

### Function: check(value)

## Class: LegacyRefResolver

### Function: validate()

### Function: retrieve(uri)

### Function: fake_urlopen(url)

### Function: handler(url)

### Function: handler(url)

### Function: handler(url)

### Function: handler(url)

### Function: handle(uri)

## Class: OhNo

### Function: resolving(this, ref)
