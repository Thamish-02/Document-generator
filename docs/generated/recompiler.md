## AI Summary

A file named recompiler.py.


## Class: GlobalExpr

## Class: FieldExpr

## Class: StructUnionExpr

## Class: EnumExpr

## Class: TypenameExpr

## Class: Recompiler

### Function: _is_file_like(maybefile)

### Function: _make_c_or_py_source(ffi, module_name, preamble, target_file, verbose)

### Function: make_c_source(ffi, module_name, preamble, target_c_file, verbose)

### Function: make_py_source(ffi, module_name, target_py_file, verbose)

### Function: _modname_to_file(outputdir, modname, extension)

### Function: _patch_meth(patchlist, cls, name, new_meth)

### Function: _unpatch_meths(patchlist)

### Function: _patch_for_embedding(patchlist)

### Function: _patch_for_target(patchlist, target)

### Function: recompile(ffi, module_name, preamble, tmpdir, call_c_compiler, c_file, source_extension, extradir, compiler_verbose, target, debug, uses_ffiplatform)

### Function: __init__(self, name, address, type_op, size, check_value)

### Function: as_c_expr(self)

### Function: as_python_expr(self)

### Function: __init__(self, name, field_offset, field_size, fbitsize, field_type_op)

### Function: as_c_expr(self)

### Function: as_python_expr(self)

### Function: as_field_python_expr(self)

### Function: __init__(self, name, type_index, flags, size, alignment, comment, first_field_index, c_fields)

### Function: as_c_expr(self)

### Function: as_python_expr(self)

### Function: __init__(self, name, type_index, size, signed, allenums)

### Function: as_c_expr(self)

### Function: as_python_expr(self)

### Function: __init__(self, name, type_index)

### Function: as_c_expr(self)

### Function: as_python_expr(self)

### Function: __init__(self, ffi, module_name, target_is_python)

### Function: needs_version(self, ver)

### Function: collect_type_table(self)

### Function: _enum_fields(self, tp)

### Function: _do_collect_type(self, tp)

### Function: _generate(self, step_name)

### Function: collect_step_tables(self)

### Function: _prnt(self, what)

### Function: write_source_to_f(self, f, preamble)

### Function: _rel_readlines(self, filename)

### Function: write_c_source_to_f(self, f, preamble)

### Function: _to_py(self, x)

### Function: write_py_source_to_f(self, f)

### Function: _gettypenum(self, type)

### Function: _convert_funcarg_to_c(self, tp, fromvar, tovar, errcode)

### Function: _extra_local_variables(self, tp, localvars, freelines)

### Function: _convert_funcarg_to_c_ptr_or_array(self, tp, fromvar, tovar, errcode)

### Function: _convert_expr_from_c(self, tp, var, context)

### Function: _typedef_type(self, tp, name)

### Function: _generate_cpy_typedef_collecttype(self, tp, name)

### Function: _generate_cpy_typedef_decl(self, tp, name)

### Function: _typedef_ctx(self, tp, name)

### Function: _generate_cpy_typedef_ctx(self, tp, name)

### Function: _generate_cpy_function_collecttype(self, tp, name)

### Function: _generate_cpy_function_decl(self, tp, name)

### Function: _generate_cpy_function_ctx(self, tp, name)

### Function: _field_type(self, tp_struct, field_name, tp_field)

### Function: _struct_collecttype(self, tp)

### Function: _struct_decl(self, tp, cname, approxname)

### Function: _struct_ctx(self, tp, cname, approxname, named_ptr)

### Function: _check_not_opaque(self, tp, location)

### Function: _add_missing_struct_unions(self)

### Function: _generate_cpy_struct_collecttype(self, tp, name)

### Function: _struct_names(self, tp)

### Function: _generate_cpy_struct_decl(self, tp, name)

### Function: _generate_cpy_struct_ctx(self, tp, name)

### Function: _generate_cpy_anonymous_collecttype(self, tp, name)

### Function: _generate_cpy_anonymous_decl(self, tp, name)

### Function: _generate_cpy_anonymous_ctx(self, tp, name)

### Function: _generate_cpy_const(self, is_int, name, tp, category, check_value)

### Function: _generate_cpy_constant_collecttype(self, tp, name)

### Function: _generate_cpy_constant_decl(self, tp, name)

### Function: _generate_cpy_constant_ctx(self, tp, name)

### Function: _generate_cpy_enum_collecttype(self, tp, name)

### Function: _generate_cpy_enum_decl(self, tp, name)

### Function: _enum_ctx(self, tp, cname)

### Function: _generate_cpy_enum_ctx(self, tp, name)

### Function: _generate_cpy_macro_collecttype(self, tp, name)

### Function: _generate_cpy_macro_decl(self, tp, name)

### Function: _generate_cpy_macro_ctx(self, tp, name)

### Function: _global_type(self, tp, global_name)

### Function: _generate_cpy_variable_collecttype(self, tp, name)

### Function: _generate_cpy_variable_decl(self, tp, name)

### Function: _generate_cpy_variable_ctx(self, tp, name)

### Function: _generate_cpy_extern_python_collecttype(self, tp, name)

### Function: _extern_python_decl(self, tp, name, tag_and_space)

### Function: _generate_cpy_extern_python_decl(self, tp, name)

### Function: _generate_cpy_dllexport_python_decl(self, tp, name)

### Function: _generate_cpy_extern_python_plus_c_decl(self, tp, name)

### Function: _generate_cpy_extern_python_ctx(self, tp, name)

### Function: _print_string_literal_in_array(self, s)

### Function: _emit_bytecode_VoidType(self, tp, index)

### Function: _emit_bytecode_PrimitiveType(self, tp, index)

### Function: _emit_bytecode_UnknownIntegerType(self, tp, index)

### Function: _emit_bytecode_UnknownFloatType(self, tp, index)

### Function: _emit_bytecode_RawFunctionType(self, tp, index)

### Function: _emit_bytecode_PointerType(self, tp, index)

### Function: _emit_bytecode_FunctionPtrType(self, tp, index)

### Function: _emit_bytecode_ArrayType(self, tp, index)

### Function: _emit_bytecode_StructType(self, tp, index)

### Function: _emit_bytecode_EnumType(self, tp, index)

## Class: NativeIO

### Function: need_indirection(type)

### Function: may_need_128_bits(tp)

### Function: write(self, s)

### Function: my_link_shared_object(self)
