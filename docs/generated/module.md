## AI Summary

A file named module.py.


## Class: _ModuleAttributeName

**Description:** For module attributes like __file__, __str__ and so on.

## Class: SubModuleDictMixin

## Class: ModuleMixin

## Class: ModuleValue

### Function: __init__(self, parent_module, string_name, string_value)

### Function: infer(self)

### Function: sub_modules_dict(self)

**Description:** Lists modules in the directory of this module (if this module is a
package).

### Function: get_filters(self, origin_scope)

### Function: py__class__(self)

### Function: is_module(self)

### Function: is_stub(self)

### Function: name(self)

### Function: _module_attributes_dict(self)

### Function: iter_star_filters(self)

### Function: star_imports(self)

### Function: get_qualified_names(self)

**Description:** A module doesn't have a qualified name, but it's important to note that
it's reachable and not `None`. With this information we can add
qualified names on top for all value children.

### Function: __init__(self, inference_state, module_node, code_lines, file_io, string_names, is_package)

### Function: is_stub(self)

### Function: py__name__(self)

### Function: py__file__(self)

**Description:** In contrast to Python's __file__ can be None.

### Function: is_package(self)

### Function: py__package__(self)

### Function: py__path__(self)

**Description:** In case of a package, this returns Python's __path__ attribute, which
is a list of paths (strings).
Returns None if the module is not a package.

### Function: _as_context(self)

### Function: __repr__(self)
