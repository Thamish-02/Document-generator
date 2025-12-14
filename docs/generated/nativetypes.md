## AI Summary

A file named nativetypes.py.


### Function: native_concat(values)

**Description:** Return a native Python type from the list of compiled nodes. If
the result is a single node, its value is returned. Otherwise, the
nodes are concatenated as strings. If the result can be parsed with
:func:`ast.literal_eval`, the parsed value is returned. Otherwise,
the string is returned.

:param values: Iterable of outputs to concatenate.

## Class: NativeCodeGenerator

**Description:** A code generator which renders Python types by not adding
``str()`` around output nodes.

## Class: NativeEnvironment

**Description:** An environment that renders templates to native Python types.

## Class: NativeTemplate

### Function: _default_finalize(value)

### Function: _output_const_repr(self, group)

### Function: _output_child_to_const(self, node, frame, finalize)

### Function: _output_child_pre(self, node, frame, finalize)

### Function: _output_child_post(self, node, frame, finalize)

### Function: render(self)

**Description:** Render the template to produce a native Python type. If the
result is a single node, its value is returned. Otherwise, the
nodes are concatenated as strings. If the result can be parsed
with :func:`ast.literal_eval`, the parsed value is returned.
Otherwise, the string is returned.
