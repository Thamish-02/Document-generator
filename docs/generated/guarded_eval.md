## AI Summary

A file named guarded_eval.py.


## Class: HasGetItem

## Class: InstancesHaveGetItem

## Class: HasGetAttr

## Class: DoesNotHaveGetAttr

### Function: _unbind_method(func)

**Description:** Get unbound method for given bound method.

Returns None if cannot get unbound method, or method is already unbound.

## Class: EvaluationPolicy

**Description:** Definition of evaluation policy.

### Function: _get_external(module_name, access_path)

**Description:** Get value from external module given a dotted access path.

Raises:
* `KeyError` if module is removed not found, and
* `AttributeError` if access path does not match an exported object

### Function: _has_original_dunder_external(value, module_name, access_path, method_name)

### Function: _has_original_dunder(value, allowed_types, allowed_methods, allowed_external, method_name)

## Class: SelectivePolicy

## Class: _DummyNamedTuple

**Description:** Used internally to retrieve methods of named tuple instance.

## Class: EvaluationContext

## Class: _IdentitySubscript

**Description:** Returns the key itself when item is requested via subscript.

## Class: GuardRejection

**Description:** Exception raised when guard rejects evaluation attempt.

### Function: guarded_eval(code, context)

**Description:** Evaluate provided code in the evaluation context.

If evaluation policy given by context is set to ``forbidden``
no evaluation will be performed; if it is set to ``dangerous``
standard :func:`eval` will be used; finally, for any other,
policy :func:`eval_node` will be called on parsed AST.

## Class: ImpersonatingDuck

**Description:** A dummy class used to create objects of other classes without calling their ``__init__``

## Class: _Duck

**Description:** A dummy class used to create objects pretending to have given attributes

### Function: _find_dunder(node_op, dunders)

### Function: eval_node(node, context)

**Description:** Evaluate AST node in provided context.

Applies evaluation restrictions defined in the context. Currently does not support evaluation of functions with keyword arguments.

Does not evaluate actions that always have side effects:

- class definitions (``class sth: ...``)
- function definitions (``def sth: ...``)
- variable assignments (``x = 1``)
- augmented assignments (``x += 1``)
- deletions (``del x``)

Does not evaluate operations which do not return values:

- assertions (``assert x``)
- pass (``pass``)
- imports (``import x``)
- control flow:

    - conditionals (``if x:``) except for ternary IfExp (``a if x else b``)
    - loops (``for`` and ``while``)
    - exception handling

The purpose of this function is to guard against unwanted side-effects;
it does not give guarantees on protection from malicious code execution.

### Function: _eval_return_type(func, node, context)

**Description:** Evaluate return type of a given callable function.

Returns the built-in type, a duck or NOT_EVALUATED sentinel.

### Function: _resolve_annotation(annotation, sig, func, node, context)

**Description:** Resolve annotation created by user with `typing` module and custom objects.

### Function: _eval_node_name(node_id, context)

### Function: _eval_or_create_duck(duck_type, node, context)

### Function: _create_duck_for_heap_type(duck_type)

**Description:** Create an imitation of an object of a given type (a duck).

Returns the duck or NOT_EVALUATED sentinel if duck could not be created.

### Function: _list_methods(cls, source)

**Description:** For use on immutable objects or with methods returning a copy

### Function: __getitem__(self, key)

### Function: __call__(self)

### Function: __getattr__(self, key)

### Function: can_get_item(self, value, item)

### Function: can_get_attr(self, value, attr)

### Function: can_operate(self, dunders, a, b)

### Function: can_call(self, func)

### Function: can_get_attr(self, value, attr)

### Function: can_get_item(self, value, item)

**Description:** Allow accessing `__getiitem__` of allow-listed instances unless it was not modified.

### Function: can_operate(self, dunders, a, b)

### Function: _operator_dunder_methods(self, dunder)

### Function: _getitem_methods(self)

### Function: _getattr_methods(self)

### Function: _getattribute_methods(self)

### Function: _safe_get_methods(self, classes, name)

### Function: __getitem__(self, key)

### Function: __init__(self, attributes, items)

### Function: __getattr__(self, attr)

### Function: __hasattr__(self, attr)

### Function: __dir__(self)

### Function: __getitem__(self, key)

### Function: __hasitem__(self, key)

### Function: _ipython_key_completions_(self)
