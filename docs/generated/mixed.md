## AI Summary

A file named mixed.py.


## Class: MixedObject

**Description:** A ``MixedObject`` is used in two ways:

1. It uses the default logic of ``parser.python.tree`` objects,
2. except for getattr calls and signatures. The names dicts are generated
   in a fashion like ``CompiledValue``.

This combined logic makes it possible to provide more powerful REPL
completion. It allows side effects that are not noticable with the default
parser structure to still be completable.

The biggest difference from CompiledValue to MixedObject is that we are
generally dealing with Python code and not with C code. This will generate
fewer special cases, because we in Python you don't have the same freedoms
to modify the runtime.

## Class: MixedContext

## Class: MixedModuleContext

## Class: MixedName

**Description:** The ``CompiledName._compiled_value`` is our MixedObject.

## Class: MixedObjectFilter

### Function: _load_module(inference_state, path)

### Function: _get_object_to_check(python_object)

**Description:** Check if inspect.getfile has a chance to find the source.

### Function: _find_syntax_node_name(inference_state, python_object)

### Function: _create(inference_state, compiled_value, module_context)

### Function: __init__(self, compiled_value, tree_value)

### Function: get_filters(self)

### Function: get_signatures(self)

### Function: py__call__(self, arguments)

### Function: get_safe_value(self, default)

### Function: array_type(self)

### Function: get_key_values(self)

### Function: py__simple_getitem__(self, index)

### Function: negate(self)

### Function: _as_context(self)

### Function: __repr__(self)

### Function: compiled_value(self)

### Function: __init__(self, wrapped_name, parent_tree_value)

### Function: start_pos(self)

### Function: infer(self)

### Function: __init__(self, inference_state, compiled_value, tree_value)

### Function: _create_name(self)
