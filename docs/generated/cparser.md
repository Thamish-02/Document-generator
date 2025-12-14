## AI Summary

A file named cparser.py.


### Function: _workaround_for_static_import_finders()

### Function: _get_parser()

### Function: _workaround_for_old_pycparser(csource)

### Function: _preprocess_extern_python(csource)

### Function: _warn_for_string_literal(csource)

### Function: _warn_for_non_extern_non_static_global_variable(decl)

### Function: _remove_line_directives(csource)

### Function: _put_back_line_directives(csource, line_directives)

### Function: _preprocess(csource)

### Function: _common_type_names(csource)

## Class: Parser

### Function: replace(m)

### Function: replace(m)

### Function: replace_keeping_newlines(m)

### Function: __init__(self)

### Function: _parse(self, csource)

### Function: _convert_pycparser_error(self, e, csource)

### Function: convert_pycparser_error(self, e, csource)

### Function: parse(self, csource, override, packed, pack, dllexport)

### Function: _internal_parse(self, csource)

### Function: _add_constants(self, key, val)

### Function: _add_integer_constant(self, name, int_str)

### Function: _process_macros(self, macros)

### Function: _declare_function(self, tp, quals, decl)

### Function: _parse_decl(self, decl)

### Function: parse_type(self, cdecl)

### Function: parse_type_and_quals(self, cdecl)

### Function: _declare(self, name, obj, included, quals)

### Function: _extract_quals(self, type)

### Function: _get_type_pointer(self, type, quals, declname)

### Function: _get_type_and_quals(self, typenode, name, partial_length_ok, typedef_example)

### Function: _parse_function_type(self, typenode, funcname)

### Function: _as_func_arg(self, type, quals)

### Function: _get_struct_union_enum_type(self, kind, type, name, nested)

### Function: _make_partial(self, tp, nested)

### Function: _parse_constant(self, exprnode, partial_length_ok)

### Function: _c_div(self, a, b)

### Function: _build_enum_type(self, explicit_name, decls)

### Function: include(self, other)

### Function: _get_unknown_type(self, decl)

### Function: _get_unknown_ptr_type(self, decl)
