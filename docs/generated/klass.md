## AI Summary

A file named klass.py.


## Class: ClassName

## Class: ClassFilter

## Class: ClassMixin

## Class: ClassValue

### Function: __init__(self, class_value, tree_name, name_context, apply_decorators)

### Function: infer(self)

### Function: api_type(self)

### Function: __init__(self, class_value, node_context, until_position, origin_scope, is_instance)

### Function: _convert_names(self, names)

### Function: _equals_origin_scope(self)

### Function: _access_possible(self, name)

### Function: _filter(self, names)

### Function: is_class(self)

### Function: is_class_mixin(self)

### Function: py__call__(self, arguments)

### Function: py__class__(self)

### Function: name(self)

### Function: py__name__(self)

### Function: py__mro__(self)

### Function: get_filters(self, origin_scope, is_instance, include_metaclasses, include_type_when_class)

### Function: get_signatures(self)

### Function: _as_context(self)

### Function: get_type_hint(self, add_class_info)

### Function: is_typeddict(self)

### Function: py__getitem__(self, index_value_set, contextualized_node)

### Function: with_generics(self, generics_tuple)

### Function: define_generics(self, type_var_dict)

### Function: list_type_vars(self)

### Function: _get_bases_arguments(self)

### Function: py__bases__(self)

### Function: get_metaclass_filters(self, metaclasses, is_instance)

### Function: get_metaclasses(self)

### Function: get_metaclass_signatures(self, metaclasses)

### Function: remap_type_vars()

**Description:** The TypeVars in the resulting classes have sometimes different names
and we need to check for that, e.g. a signature can be:

def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

However, the iterator is defined as Iterator[_T_co], which means it has
a different type var name.
