## AI Summary

A file named arguments.py.


### Function: try_iter_content(types, depth)

**Description:** Helper method for static analysis.

## Class: ParamIssue

### Function: repack_with_argument_clinic(clinic_string)

**Description:** Transforms a function or method with arguments to the signature that is
given as an argument clinic notation.

Argument clinic is part of CPython and used for all the functions that are
implemented in C (Python 3.7):

    str.split.__text_signature__
    # Results in: '($self, /, sep=None, maxsplit=-1)'

### Function: iterate_argument_clinic(inference_state, arguments, clinic_string)

**Description:** Uses a list with argument clinic information (see PEP 436).

### Function: _parse_argument_clinic(string)

## Class: _AbstractArgumentsMixin

## Class: AbstractArguments

### Function: unpack_arglist(arglist)

## Class: TreeArguments

## Class: ValuesArguments

## Class: TreeArgumentsWrapper

### Function: _iterate_star_args(context, array, input_node, funcdef)

### Function: _star_star_dict(context, array, input_node, funcdef)

### Function: decorator(func)

### Function: unpack(self, funcdef)

### Function: get_calling_nodes(self)

### Function: __init__(self, inference_state, context, argument_node, trailer)

**Description:** :param argument_node: May be an argument_node or a list of nodes.

### Function: create_cached(cls)

### Function: unpack(self, funcdef)

### Function: _as_tree_tuple_objects(self)

### Function: iter_calling_names_with_star(self)

### Function: __repr__(self)

### Function: get_calling_nodes(self)

### Function: __init__(self, values_list)

### Function: unpack(self, funcdef)

### Function: __repr__(self)

### Function: __init__(self, arguments)

### Function: context(self)

### Function: argument_node(self)

### Function: trailer(self)

### Function: unpack(self, func)

### Function: get_calling_nodes(self)

### Function: __repr__(self)

### Function: wrapper(value, arguments)
