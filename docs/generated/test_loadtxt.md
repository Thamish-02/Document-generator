## AI Summary

A file named test_loadtxt.py.


### Function: test_scientific_notation()

**Description:** Test that both 'e' and 'E' are parsed correctly.

### Function: test_comment_multiple_chars(comment)

### Function: mixed_types_structured()

**Description:** Fixture providing heterogeneous input data with a structured dtype, along
with the associated structured array.

### Function: test_structured_dtype_and_skiprows_no_empty_lines(skiprows, mixed_types_structured)

### Function: test_unpack_structured(mixed_types_structured)

### Function: test_structured_dtype_with_shape()

### Function: test_structured_dtype_with_multi_shape()

### Function: test_nested_structured_subarray()

### Function: test_structured_dtype_offsets()

### Function: test_exception_negative_row_limits(param)

**Description:** skiprows and max_rows should raise for negative parameters.

### Function: test_exception_noninteger_row_limits(param)

### Function: test_ndmin_single_row_or_col(data, shape)

### Function: test_bad_ndmin(badval)

### Function: test_blank_lines_spaces_delimit(ws)

### Function: test_blank_lines_normal_delimiter()

### Function: test_maxrows_no_blank_lines(dtype)

### Function: test_exception_message_bad_values(dtype)

### Function: test_converters_negative_indices()

### Function: test_converters_negative_indices_with_usecols()

### Function: test_ragged_error()

### Function: test_ragged_usecols()

### Function: test_empty_usecols()

### Function: test_large_unicode_characters(c1, c2)

### Function: test_unicode_with_converter()

### Function: test_converter_with_structured_dtype()

### Function: test_converter_with_unicode_dtype()

**Description:** With the 'bytes' encoding, tokens are encoded prior to being
passed to the converter. This means that the output of the converter may
be bytes instead of unicode as expected by `read_rows`.

This test checks that outputs from the above scenario are properly decoded
prior to parsing by `read_rows`.

### Function: test_read_huge_row()

### Function: test_huge_float(dtype)

### Function: test_string_no_length_given(given_dtype, expected_dtype)

**Description:** The given dtype is just 'S' or 'U' with no length. In these cases, the
length of the resulting dtype is determined by the longest string found
in the file.

### Function: test_float_conversion()

**Description:** Some tests that the conversion to float64 works as accurately as the
Python built-in `float` function. In a naive version of the float parser,
these strings resulted in values that were off by an ULP or two.

### Function: test_bool()

### Function: test_integer_signs(dtype)

### Function: test_implicit_cast_float_to_int_fails(dtype)

### Function: test_complex_parsing(dtype, with_parens)

### Function: test_read_from_generator()

### Function: test_read_from_generator_multitype()

### Function: test_read_from_bad_generator()

### Function: test_object_cleanup_on_read_error()

### Function: test_character_not_bytes_compatible()

**Description:** Test exception when a character cannot be encoded as 'S'.

### Function: test_invalid_converter(conv)

### Function: test_converters_dict_raises_non_integer_key()

### Function: test_converters_dict_raises_non_col_key(bad_col_ind)

### Function: test_converters_dict_raises_val_not_callable()

### Function: test_quoted_field(q)

### Function: test_quoted_field_with_whitepace_delimiter(q)

### Function: test_quote_support_default()

**Description:** Support for quoted fields is disabled by default.

### Function: test_quotechar_multichar_error()

### Function: test_comment_multichar_error_with_quote()

### Function: test_structured_dtype_with_quotes()

### Function: test_quoted_field_is_not_empty()

### Function: test_quoted_field_is_not_empty_nonstrict()

### Function: test_consecutive_quotechar_escaped()

### Function: test_warn_on_no_data(data, ndmin, usecols)

**Description:** Check that a UserWarning is emitted when no data is read from input.

### Function: test_warn_on_skipped_data(skiprows)

### Function: test_byteswapping_and_unaligned(dtype, value, swap)

### Function: test_unicode_whitespace_stripping(dtype)

### Function: test_unicode_whitespace_stripping_complex(dtype)

### Function: test_bad_complex(dtype, field)

### Function: test_nul_character_error(dtype)

### Function: test_no_thousands_support(dtype)

### Function: test_bad_newline_in_iterator(data)

### Function: test_good_newline_in_iterator(data)

### Function: test_universal_newlines_quoted(newline)

### Function: test_null_character()

### Function: test_iterator_fails_getting_next_line()

## Class: TestCReaderUnitTests

### Function: test_delimiter_comment_collision_raises()

### Function: test_delimiter_quotechar_collision_raises()

### Function: test_comment_quotechar_collision_raises()

### Function: test_delimiter_and_multiple_comments_collision_raises()

### Function: test_collision_with_default_delimiter_raises(ws)

### Function: test_control_character_newline_raises(nl)

### Function: test_parametric_unit_discovery(generic_data, long_datum, unitless_dtype, expected_dtype, nrows)

**Description:** Check that the correct unit (e.g. month, day, second) is discovered from
the data when a user specifies a unitless datetime.

### Function: test_str_dtype_unit_discovery_with_converter()

### Function: test_control_character_empty()

### Function: test_control_characters_as_bytes()

**Description:** Byte control characters (comments, delimiter) are supported.

### Function: test_field_growing_cases()

### Function: test_maxrows_exceeding_chunksize(nmax)

### Function: test_skiprow_exceeding_maxrows_exceeding_chunksize(tmpdir, nskip)

### Function: gen()

### Function: gen()

### Function: gen()

### Function: conv(x)

## Class: BadSequence

### Function: test_not_an_filelike(self)

### Function: test_filelike_read_fails(self)

### Function: test_filelike_bad_read(self)

### Function: test_not_an_iter(self)

### Function: test_bad_type(self)

### Function: test_bad_encoding(self)

### Function: test_manual_universal_newlines(self, newline)

### Function: __len__(self)

### Function: __getitem__(self, item)

## Class: BadFileLike

## Class: BadFileLike

### Function: read(self, size)

### Function: read(self, size)
