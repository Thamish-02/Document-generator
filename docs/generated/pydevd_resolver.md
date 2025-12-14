## AI Summary

A file named pydevd_resolver.py.


## Class: UnableToResolveVariableException

### Function: sorted_attributes_key(attr_name)

## Class: DefaultResolver

**Description:** DefaultResolver is the class that'll actually resolve how to show some variable.

## Class: DAPGrouperResolver

### Function: _does_obj_repr_evaluate_to_obj(obj)

**Description:** If obj is an object where evaluating its representation leads to
the same object, return True, otherwise, return False.

## Class: DictResolver

### Function: _apply_evaluate_name(parent_name, evaluate_name)

## Class: MoreItemsRange

## Class: MoreItems

## Class: ForwardInternalResolverToObject

**Description:** To be used when we provide some internal object that'll actually do the resolution.

## Class: TupleResolver

## Class: SetResolver

**Description:** Resolves a set as dict id(object)->object

## Class: InstanceResolver

## Class: JyArrayResolver

**Description:** This resolves a regular Object[] array from java

## Class: MultiValueDictResolver

## Class: DjangoFormResolver

## Class: DequeResolver

## Class: OrderedDictResolver

## Class: FrameResolver

**Description:** This resolves a frame.

## Class: InspectStub

### Function: get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values)

### Function: resolve(self, var, attribute)

### Function: get_contents_debug_adapter_protocol(self, obj, fmt)

### Function: get_dictionary(self, var, names, used___dict__)

### Function: _get_jy_dictionary(self, obj)

### Function: get_names(self, var)

### Function: _get_py_dictionary(self, var, names, used___dict__)

**Description:** :return tuple(names, used___dict__), where used___dict__ means we have to access
using obj.__dict__[name] instead of getattr(obj, name)

### Function: get_contents_debug_adapter_protocol(self, obj, fmt)

### Function: resolve(self, dct, key)

### Function: key_to_str(self, key, fmt)

### Function: init_dict(self)

### Function: get_contents_debug_adapter_protocol(self, dct, fmt)

**Description:** This method is to be used in the case where the variables are all saved by its id (and as
such don't need to have the `resolve` method called later on, so, keys don't need to
embed the reference in the key).

Note that the return should be ordered.

:return list(tuple(name:str, value:object, evaluateName:str))

### Function: get_dictionary(self, dct)

### Function: __init__(self, value, from_i, to_i)

### Function: get_contents_debug_adapter_protocol(self, _self, fmt)

### Function: get_dictionary(self, _self, fmt)

### Function: resolve(self, attribute)

**Description:** :param var: that's the original object we're dealing with.
:param attribute: that's the key to resolve
    -- either the dict key in get_dictionary or the name in the dap protocol.

### Function: __eq__(self, o)

### Function: __str__(self)

### Function: __init__(self, value, handled_items)

### Function: get_contents_debug_adapter_protocol(self, _self, fmt)

### Function: get_dictionary(self, _self, fmt)

### Function: resolve(self, attribute)

### Function: __eq__(self, o)

### Function: __str__(self)

### Function: get_contents_debug_adapter_protocol(self, obj, fmt)

### Function: get_dictionary(self, var, fmt)

### Function: resolve(self, var, attribute)

### Function: resolve(self, var, attribute)

**Description:** :param var: that's the original object we're dealing with.
:param attribute: that's the key to resolve
    -- either the dict key in get_dictionary or the name in the dap protocol.

### Function: get_contents_debug_adapter_protocol(self, lst, fmt)

**Description:** This method is to be used in the case where the variables are all saved by its id (and as
such don't need to have the `resolve` method called later on, so, keys don't need to
embed the reference in the key).

Note that the return should be ordered.

:return list(tuple(name:str, value:object, evaluateName:str))

### Function: get_dictionary(self, var, fmt)

### Function: get_contents_debug_adapter_protocol(self, obj, fmt)

### Function: resolve(self, var, attribute)

### Function: get_dictionary(self, var)

### Function: change_var_from_name(self, container, name, new_value)

### Function: resolve(self, var, attribute)

### Function: get_dictionary(self, obj)

### Function: resolve(self, var, attribute)

### Function: get_dictionary(self, obj)

### Function: resolve(self, dct, key)

### Function: get_dictionary(self, var, names)

### Function: get_dictionary(self, var)

### Function: init_dict(self)

### Function: resolve(self, obj, attribute)

### Function: get_dictionary(self, obj)

### Function: get_frame_stack(self, frame)

### Function: get_frame_name(self, frame)

### Function: isbuiltin(self, _args)

### Function: isroutine(self, object)
