## AI Summary

A file named test_cli.py.


### Function: fake_validator()

### Function: fake_open(all_contents)

### Function: _message_for(non_json)

## Class: TestCLI

## Class: TestParser

## Class: TestCLIIntegration

## Class: FakeValidator

### Function: open(path)

### Function: run_cli(self, argv, files, stdin, exit_code)

### Function: assertOutputs(self, stdout, stderr)

### Function: test_invalid_instance(self)

### Function: test_invalid_instance_pretty_output(self)

### Function: test_invalid_instance_explicit_plain_output(self)

### Function: test_invalid_instance_multiple_errors(self)

### Function: test_invalid_instance_multiple_errors_pretty_output(self)

### Function: test_multiple_invalid_instances(self)

### Function: test_multiple_invalid_instances_pretty_output(self)

### Function: test_custom_error_format(self)

### Function: test_invalid_schema(self)

### Function: test_invalid_schema_pretty_output(self)

### Function: test_invalid_schema_multiple_errors(self)

### Function: test_invalid_schema_multiple_errors_pretty_output(self)

### Function: test_invalid_schema_with_invalid_instance(self)

**Description:** "Validating" an instance that's invalid under an invalid schema
just shows the schema error.

### Function: test_invalid_schema_with_invalid_instance_pretty_output(self)

### Function: test_invalid_instance_continues_with_the_rest(self)

### Function: test_custom_error_format_applies_to_schema_errors(self)

### Function: test_instance_is_invalid_JSON(self)

### Function: test_instance_is_invalid_JSON_pretty_output(self)

### Function: test_instance_is_invalid_JSON_on_stdin(self)

### Function: test_instance_is_invalid_JSON_on_stdin_pretty_output(self)

### Function: test_schema_is_invalid_JSON(self)

### Function: test_schema_is_invalid_JSON_pretty_output(self)

### Function: test_schema_and_instance_are_both_invalid_JSON(self)

**Description:** Only the schema error is reported, as we abort immediately.

### Function: test_schema_and_instance_are_both_invalid_JSON_pretty_output(self)

**Description:** Only the schema error is reported, as we abort immediately.

### Function: test_instance_does_not_exist(self)

### Function: test_instance_does_not_exist_pretty_output(self)

### Function: test_schema_does_not_exist(self)

### Function: test_schema_does_not_exist_pretty_output(self)

### Function: test_neither_instance_nor_schema_exist(self)

### Function: test_neither_instance_nor_schema_exist_pretty_output(self)

### Function: test_successful_validation(self)

### Function: test_successful_validation_pretty_output(self)

### Function: test_successful_validation_of_stdin(self)

### Function: test_successful_validation_of_stdin_pretty_output(self)

### Function: test_successful_validation_of_just_the_schema(self)

### Function: test_successful_validation_of_just_the_schema_pretty_output(self)

### Function: test_successful_validation_via_explicit_base_uri(self)

### Function: test_unsuccessful_validation_via_explicit_base_uri(self)

### Function: test_nonexistent_file_with_explicit_base_uri(self)

### Function: test_invalid_explicit_base_uri(self)

### Function: test_it_validates_using_the_latest_validator_when_unspecified(self)

### Function: test_it_validates_using_draft7_when_specified(self)

**Description:** Specifically, `const` validation applies for Draft 7.

### Function: test_it_validates_using_draft4_when_specified(self)

**Description:** Specifically, `const` validation *does not* apply for Draft 4.

### Function: test_find_validator_by_fully_qualified_object_name(self)

### Function: test_find_validator_in_jsonschema(self)

### Function: cli_output_for(self)

### Function: test_unknown_output(self)

### Function: test_useless_error_format(self)

### Function: test_license(self)

### Function: test_version(self)

### Function: test_no_arguments_shows_usage_notes(self)

### Function: __init__(self)

### Function: iter_errors(self, instance)

### Function: check_schema(self, schema)
