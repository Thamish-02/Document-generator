## AI Summary

A file named vengine_cpy.py.


## Class: VCPythonEngine

### Function: __init__(self, verifier)

### Function: patch_extension_kwds(self, kwds)

### Function: find_module(self, module_name, path, so_suffixes)

### Function: collect_types(self)

### Function: _prnt(self, what)

### Function: _gettypenum(self, type)

### Function: _do_collect_type(self, tp)

### Function: write_source_to_f(self)

### Function: load_library(self, flags)

### Function: _get_declarations(self)

### Function: _generate(self, step_name)

### Function: _load(self, module, step_name)

### Function: _generate_nothing(self, tp, name)

### Function: _loaded_noop(self, tp, name, module)

### Function: _convert_funcarg_to_c(self, tp, fromvar, tovar, errcode)

### Function: _extra_local_variables(self, tp, localvars, freelines)

### Function: _convert_funcarg_to_c_ptr_or_array(self, tp, fromvar, tovar, errcode)

### Function: _convert_expr_from_c(self, tp, var, context)

### Function: _generate_cpy_function_collecttype(self, tp, name)

### Function: _generate_cpy_function_decl(self, tp, name)

### Function: _generate_cpy_function_method(self, tp, name)

### Function: _loaded_cpy_function(self, tp, name, module, library)

### Function: _generate_cpy_struct_decl(self, tp, name)

### Function: _generate_cpy_struct_method(self, tp, name)

### Function: _loading_cpy_struct(self, tp, name, module)

### Function: _loaded_cpy_struct(self, tp, name, module)

### Function: _generate_cpy_union_decl(self, tp, name)

### Function: _generate_cpy_union_method(self, tp, name)

### Function: _loading_cpy_union(self, tp, name, module)

### Function: _loaded_cpy_union(self, tp, name, module)

### Function: _generate_struct_or_union_decl(self, tp, prefix, name)

### Function: _generate_struct_or_union_method(self, tp, prefix, name)

### Function: _loading_struct_or_union(self, tp, prefix, name, module)

### Function: _loaded_struct_or_union(self, tp)

### Function: _generate_cpy_anonymous_decl(self, tp, name)

### Function: _generate_cpy_anonymous_method(self, tp, name)

### Function: _loading_cpy_anonymous(self, tp, name, module)

### Function: _loaded_cpy_anonymous(self, tp, name, module)

### Function: _generate_cpy_const(self, is_int, name, tp, category, vartp, delayed, size_too, check_value)

### Function: _generate_cpy_constant_collecttype(self, tp, name)

### Function: _generate_cpy_constant_decl(self, tp, name)

### Function: _check_int_constant_value(self, name, value, err_prefix)

### Function: _enum_funcname(self, prefix, name)

### Function: _generate_cpy_enum_decl(self, tp, name, prefix)

### Function: _loading_cpy_enum(self, tp, name, module)

### Function: _loaded_cpy_enum(self, tp, name, module, library)

### Function: _generate_cpy_macro_decl(self, tp, name)

### Function: _generate_cpy_variable_collecttype(self, tp, name)

### Function: _generate_cpy_variable_decl(self, tp, name)

### Function: _loaded_cpy_variable(self, tp, name, module, library)

### Function: _generate_setup_custom(self)

## Class: FFILibrary

### Function: getter(library)

### Function: setter(library, value)

### Function: __dir__(self)

### Function: check(realvalue, expectedvalue, msg)
