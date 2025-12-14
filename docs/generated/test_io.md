## AI Summary

A file named test_io.py.


## Class: TextIO

**Description:** Helper IO class.

Writes encode strings to bytes if needed, reads return bytes.
This makes it easier to emulate files opened in binary mode
without needing to explicitly convert strings to bytes in
setting up the test data.

### Function: strptime(s, fmt)

**Description:** This function is available in the datetime module only from Python >=
2.5.

## Class: RoundtripTest

## Class: TestSaveLoad

## Class: TestSavezLoad

## Class: TestSaveTxt

## Class: LoadTxtBase

## Class: TestLoadTxt

## Class: Testfromregex

## Class: TestFromTxt

## Class: TestPathUsage

### Function: test_gzip_load()

## Class: JustWriter

## Class: JustReader

### Function: test_ducktyping()

### Function: test_gzip_loadtxt()

### Function: test_gzip_loadtxt_from_string()

### Function: test_npzfile_dict()

### Function: test_load_refcount()

### Function: test_load_multiple_arrays_until_eof()

### Function: test_savez_nopickle()

### Function: __init__(self, s)

### Function: write(self, s)

### Function: writelines(self, lines)

### Function: roundtrip(self, save_func)

**Description:** save_func : callable
    Function used to save arrays to file.
file_on_disk : bool
    If true, store the file on disk, instead of in a
    string buffer.
save_kwds : dict
    Parameters passed to `save_func`.
load_kwds : dict
    Parameters passed to `numpy.load`.
args : tuple of arrays
    Arrays stored to file.

### Function: check_roundtrips(self, a)

### Function: test_array(self)

### Function: test_array_object(self)

### Function: test_1D(self)

### Function: test_mmap(self)

### Function: test_record(self)

### Function: test_format_2_0(self)

### Function: roundtrip(self)

### Function: roundtrip(self)

### Function: test_big_arrays(self)

### Function: test_multiple_arrays(self)

### Function: test_named_arrays(self)

### Function: test_tuple_getitem_raises(self)

### Function: test_BagObj(self)

### Function: test_savez_filename_clashes(self)

### Function: test_not_closing_opened_fid(self)

### Function: test_closing_fid(self)

### Function: test_closing_zipfile_after_load(self)

### Function: test_repr_lists_keys(self, count, expected_repr)

### Function: test_array(self)

### Function: test_1D(self)

### Function: test_0D_3D(self)

### Function: test_structured(self)

### Function: test_structured_padded(self)

### Function: test_multifield_view(self)

### Function: test_delimiter(self)

### Function: test_format(self)

### Function: test_header_footer(self)

### Function: test_file_roundtrip(self, filename_type)

### Function: test_complex_arrays(self)

### Function: test_complex_negative_exponent(self)

### Function: test_custom_writer(self)

### Function: test_unicode(self)

### Function: test_unicode_roundtrip(self)

### Function: test_unicode_bytestream(self)

### Function: test_unicode_stringstream(self)

### Function: test_unicode_and_bytes_fmt(self, iotype)

### Function: test_large_zip(self)

### Function: check_compressed(self, fopen, suffixes)

### Function: test_compressed_gzip(self)

### Function: test_compressed_bz2(self)

### Function: test_compressed_lzma(self)

### Function: test_encoding(self)

### Function: test_stringload(self)

### Function: test_binary_decode(self)

### Function: test_converters_decode(self)

### Function: test_converters_nodecode(self)

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: test_record(self)

### Function: test_array(self)

### Function: test_1D(self)

### Function: test_missing(self)

### Function: test_converters_with_usecols(self)

### Function: test_comments_unicode(self)

### Function: test_comments_byte(self)

### Function: test_comments_multiple(self)

### Function: test_comments_multi_chars(self)

### Function: test_skiprows(self)

### Function: test_usecols(self)

### Function: test_bad_usecols(self)

### Function: test_fancy_dtype(self)

### Function: test_shaped_dtype(self)

### Function: test_3d_shaped_dtype(self)

### Function: test_str_dtype(self)

### Function: test_empty_file(self)

### Function: test_unused_converter(self)

### Function: test_dtype_with_object(self)

### Function: test_uint64_type(self)

### Function: test_int64_type(self)

### Function: test_from_float_hex(self)

### Function: test_default_float_converter_no_default_hex_conversion(self)

**Description:** Ensure that fromhex is only used for values with the correct prefix and
is not called by default. Regression test related to gh-19598.

### Function: test_default_float_converter_exception(self)

**Description:** Ensure that the exception message raised during failed floating point
conversion is correct. Regression test related to gh-19598.

### Function: test_from_complex(self)

### Function: test_complex_misformatted(self)

### Function: test_universal_newline(self)

### Function: test_empty_field_after_tab(self)

### Function: test_unpack_structured(self)

### Function: test_ndmin_keyword(self)

### Function: test_generator_source(self)

### Function: test_bad_line(self)

### Function: test_none_as_string(self)

### Function: test_binary_load(self)

### Function: test_max_rows(self)

### Function: test_max_rows_with_skiprows(self)

### Function: test_max_rows_with_read_continuation(self)

### Function: test_max_rows_larger(self)

### Function: test_max_rows_empty_lines(self, skip, data)

### Function: test_record(self)

### Function: test_record_2(self)

### Function: test_record_3(self)

### Function: test_record_unicode(self, path_type)

### Function: test_compiled_bytes(self)

### Function: test_bad_dtype_not_structured(self)

### Function: test_record(self)

### Function: test_array(self)

### Function: test_1D(self)

### Function: test_comments(self)

### Function: test_skiprows(self)

### Function: test_skip_footer(self)

### Function: test_skip_footer_with_invalid(self)

### Function: test_header(self)

### Function: test_auto_dtype(self)

### Function: test_auto_dtype_uniform(self)

### Function: test_fancy_dtype(self)

### Function: test_names_overwrite(self)

### Function: test_bad_fname(self)

### Function: test_commented_header(self)

### Function: test_names_and_comments_none(self)

### Function: test_file_is_closed_on_error(self)

### Function: test_autonames_and_usecols(self)

### Function: test_converters_with_usecols(self)

### Function: test_converters_with_usecols_and_names(self)

### Function: test_converters_cornercases(self)

### Function: test_converters_cornercases2(self)

### Function: test_unused_converter(self)

### Function: test_invalid_converter(self)

### Function: test_tricky_converter_bug1666(self)

### Function: test_dtype_with_converters(self)

### Function: test_dtype_with_converters_and_usecols(self)

### Function: test_dtype_with_object(self)

### Function: test_dtype_with_object_no_converter(self)

### Function: test_userconverters_with_explicit_dtype(self)

### Function: test_utf8_userconverters_with_explicit_dtype(self)

### Function: test_spacedelimiter(self)

### Function: test_integer_delimiter(self)

### Function: test_missing(self)

### Function: test_missing_with_tabs(self)

### Function: test_usecols(self)

### Function: test_usecols_as_css(self)

### Function: test_usecols_with_structured_dtype(self)

### Function: test_usecols_with_integer(self)

### Function: test_usecols_with_named_columns(self)

### Function: test_empty_file(self)

### Function: test_fancy_dtype_alt(self)

### Function: test_shaped_dtype(self)

### Function: test_withmissing(self)

### Function: test_user_missing_values(self)

### Function: test_user_filling_values(self)

### Function: test_withmissing_float(self)

### Function: test_with_masked_column_uniform(self)

### Function: test_with_masked_column_various(self)

### Function: test_invalid_raise(self)

### Function: test_invalid_raise_with_usecols(self)

### Function: test_inconsistent_dtype(self)

### Function: test_default_field_format(self)

### Function: test_single_dtype_wo_names(self)

### Function: test_single_dtype_w_explicit_names(self)

### Function: test_single_dtype_w_implicit_names(self)

### Function: test_easy_structured_dtype(self)

### Function: test_autostrip(self)

### Function: test_replace_space(self)

### Function: test_replace_space_known_dtype(self)

### Function: test_incomplete_names(self)

### Function: test_names_auto_completion(self)

### Function: test_names_with_usecols_bug1636(self)

### Function: test_fixed_width_names(self)

### Function: test_filling_values(self)

### Function: test_comments_is_none(self)

### Function: test_latin1(self)

### Function: test_binary_decode_autodtype(self)

### Function: test_utf8_byte_encoding(self)

### Function: test_utf8_file(self)

### Function: test_utf8_file_nodtype_unicode(self)

### Function: test_recfromtxt(self)

### Function: test_recfromcsv(self)

### Function: test_max_rows(self)

### Function: test_gft_using_filename(self)

### Function: test_gft_from_gzip(self)

### Function: test_gft_using_generator(self)

### Function: test_auto_dtype_largeint(self)

### Function: test_unpack_float_data(self)

### Function: test_unpack_structured(self)

### Function: test_unpack_auto_dtype(self)

### Function: test_unpack_single_name(self)

### Function: test_squeeze_scalar(self)

### Function: test_ndmin_keyword(self, ndim)

### Function: test_loadtxt(self)

### Function: test_save_load(self)

### Function: test_save_load_memmap(self)

### Function: test_save_load_memmap_readwrite(self, filename_type)

### Function: test_savez_load(self, filename_type)

### Function: test_savez_compressed_load(self, filename_type)

### Function: test_genfromtxt(self, filename_type)

### Function: test_recfromtxt(self, filename_type)

### Function: test_recfromcsv(self, filename_type)

### Function: __init__(self, base)

### Function: write(self, s)

### Function: flush(self)

### Function: __init__(self, base)

### Function: read(self, n)

### Function: seek(self, off, whence)

### Function: writer(error_list)

## Class: CustomWriter

### Function: check_large_zip(memoryerror_raised)

## Class: CrazyInt

### Function: count()

### Function: f()

### Function: f()

### Function: count()

### Function: write(self, text)

### Function: __index__(self)
