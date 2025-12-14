## AI Summary

A file named vengine_gen.py.


## Class: VGenericEngine

### Function: __init__(self, verifier)

### Function: patch_extension_kwds(self, kwds)

### Function: find_module(self, module_name, path, so_suffixes)

### Function: collect_types(self)

### Function: _prnt(self, what)

### Function: write_source_to_f(self)

### Function: load_library(self, flags)

### Function: _get_declarations(self)

### Function: _generate(self, step_name)

### Function: _load(self, module, step_name)

### Function: _generate_nothing(self, tp, name)

### Function: _loaded_noop(self, tp, name, module)

### Function: _generate_gen_function_decl(self, tp, name)

### Function: _loaded_gen_function(self, tp, name, module, library)

### Function: _make_struct_wrapper(self, oldfunc, i, tp, base_tp)

### Function: _generate_gen_struct_decl(self, tp, name)

### Function: _loading_gen_struct(self, tp, name, module)

### Function: _loaded_gen_struct(self, tp, name, module)

### Function: _generate_gen_union_decl(self, tp, name)

### Function: _loading_gen_union(self, tp, name, module)

### Function: _loaded_gen_union(self, tp, name, module)

### Function: _generate_struct_or_union_decl(self, tp, prefix, name)

### Function: _loading_struct_or_union(self, tp, prefix, name, module)

### Function: _loaded_struct_or_union(self, tp)

### Function: _generate_gen_anonymous_decl(self, tp, name)

### Function: _loading_gen_anonymous(self, tp, name, module)

### Function: _loaded_gen_anonymous(self, tp, name, module)

### Function: _generate_gen_const(self, is_int, name, tp, category, check_value)

### Function: _generate_gen_constant_decl(self, tp, name)

### Function: _load_constant(self, is_int, tp, name, module, check_value)

### Function: _loaded_gen_constant(self, tp, name, module, library)

### Function: _check_int_constant_value(self, name, value)

### Function: _load_known_int_constant(self, module, funcname)

### Function: _enum_funcname(self, prefix, name)

### Function: _generate_gen_enum_decl(self, tp, name, prefix)

### Function: _loading_gen_enum(self, tp, name, module, prefix)

### Function: _loaded_gen_enum(self, tp, name, module, library)

### Function: _generate_gen_macro_decl(self, tp, name)

### Function: _loaded_gen_macro(self, tp, name, module, library)

### Function: _generate_gen_variable_decl(self, tp, name)

### Function: _loaded_gen_variable(self, tp, name, module, library)

## Class: FFILibrary

### Function: getter(library)

### Function: setter(library, value)

### Function: __dir__(self)

### Function: newfunc()

### Function: newfunc()

### Function: check(realvalue, expectedvalue, msg)
