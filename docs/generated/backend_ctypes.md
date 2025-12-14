## AI Summary

A file named backend_ctypes.py.


## Class: CTypesType

## Class: CTypesData

## Class: CTypesGenericPrimitive

## Class: CTypesGenericArray

## Class: CTypesGenericPtr

## Class: CTypesBaseStructOrUnion

## Class: CTypesBackend

## Class: CTypesLibrary

### Function: __init__(self)

### Function: _newp(cls, init)

### Function: _to_ctypes(value)

### Function: _arg_to_ctypes(cls)

### Function: _create_ctype_obj(cls, init)

### Function: _from_ctypes(ctypes_value)

### Function: _get_c_name(cls, replace_with)

### Function: _fix_class(cls)

### Function: _get_own_repr(self)

### Function: _addr_repr(self, address)

### Function: __repr__(self, c_name)

### Function: _convert_to_address(self, BClass)

### Function: _get_size(cls)

### Function: _get_size_of_instance(self)

### Function: _cast_from(cls, source)

### Function: _cast_to_integer(self)

### Function: _alignment(cls)

### Function: __iter__(self)

### Function: _make_cmp(name)

### Function: __hash__(self)

### Function: _to_string(self, maxlen)

### Function: __hash__(self)

### Function: _get_own_repr(self)

### Function: _newp(cls, init)

### Function: __iter__(self)

### Function: _get_own_repr(self)

### Function: _newp(cls, init)

### Function: _cast_from(cls, source)

### Function: _new_pointer_at(cls, address)

### Function: _get_own_repr(self)

### Function: _cast_to_integer(self)

### Function: __nonzero__(self)

### Function: _to_ctypes(cls, value)

### Function: _from_ctypes(cls, ctypes_ptr)

### Function: _initialize(cls, ctypes_ptr, value)

### Function: _convert_to_address(self, BClass)

### Function: _create_ctype_obj(cls, init)

### Function: _get_own_repr(self)

### Function: _offsetof(cls, fieldname)

### Function: _convert_to_address(self, BClass)

### Function: _from_ctypes(cls, ctypes_struct_or_union)

### Function: _to_ctypes(cls, value)

### Function: __repr__(self, c_name)

### Function: __init__(self)

### Function: set_ffi(self, ffi)

### Function: _get_types(self)

### Function: load_library(self, path, flags)

### Function: new_void_type(self)

### Function: new_primitive_type(self, name)

### Function: new_pointer_type(self, BItem)

### Function: new_array_type(self, CTypesPtr, length)

### Function: _new_struct_or_union(self, kind, name, base_ctypes_class)

### Function: new_struct_type(self, name)

### Function: new_union_type(self, name)

### Function: complete_struct_or_union(self, CTypesStructOrUnion, fields, tp, totalsize, totalalignment, sflags, pack)

### Function: new_function_type(self, BArgs, BResult, has_varargs)

### Function: new_enum_type(self, name, enumerators, enumvalues, CTypesInt)

### Function: get_errno(self)

### Function: set_errno(self, value)

### Function: string(self, b, maxlen)

### Function: buffer(self, bptr, size)

### Function: sizeof(self, cdata_or_BType)

### Function: alignof(self, BType)

### Function: newp(self, BType, source)

### Function: cast(self, BType, source)

### Function: callback(self, BType, source, error, onerror)

### Function: gcp(self, cdata, destructor, size)

### Function: getcname(self, BType, replace_with)

### Function: typeoffsetof(self, BType, fieldname, num)

### Function: rawaddressof(self, BTypePtr, cdata, offset)

### Function: __init__(self, backend, cdll)

### Function: load_function(self, BType, name)

### Function: read_variable(self, BType, name)

### Function: write_variable(self, BType, name, value)

### Function: cmp(self, other)

## Class: CTypesVoid

### Function: _cast_source_to_int(source)

## Class: CTypesPrimitive

## Class: CTypesPtr

## Class: CTypesArray

## Class: struct_or_union

## Class: CTypesStructOrUnion

### Function: _create_ctype_obj(init)

### Function: initialize(blob, init)

## Class: CTypesFunctionPtr

## Class: CTypesEnum

### Function: remove(k)

### Function: _from_ctypes(novalue)

### Function: _to_ctypes(novalue)

### Function: __init__(self, value)

### Function: _create_ctype_obj(init)

### Function: _from_ctypes(value)

### Function: _initialize(blob, init)

### Function: __init__(self, init)

### Function: __add__(self, other)

### Function: __sub__(self, other)

### Function: __getitem__(self, index)

### Function: __setitem__(self, index, value)

### Function: _get_own_repr(self)

### Function: __init__(self, init)

### Function: _initialize(blob, init)

### Function: __len__(self)

### Function: __getitem__(self, index)

### Function: __setitem__(self, index, value)

### Function: _get_own_repr(self)

### Function: _convert_to_address(self, BClass)

### Function: _from_ctypes(ctypes_array)

### Function: _arg_to_ctypes(value)

### Function: __add__(self, other)

### Function: _cast_from(cls, source)

### Function: getter(self, fname)

### Function: setter(self, value, fname)

### Function: __init__(self, init, error)

### Function: _initialize(ctypes_ptr, value)

### Function: __repr__(self)

### Function: _get_own_repr(self)

### Function: __call__(self)

### Function: _get_own_repr(self)

### Function: _to_string(self, maxlen)

## Class: MyRef

### Function: _cast_from(cls, source)

### Function: __int__(self)

### Function: _cast_from(cls, source)

### Function: __int__(self)

### Function: _cast_from(cls, source)

### Function: __int__(self)

### Function: _cast_from(cls, source)

### Function: __int__(self)

### Function: __float__(self)

### Function: _to_ctypes(x)

### Function: _to_ctypes(x)

### Function: __nonzero__(self)

### Function: __nonzero__(self)

### Function: _to_ctypes(x)

### Function: _to_string(self, maxlen)

### Function: _to_string(self, maxlen)

### Function: _arg_to_ctypes(cls)

### Function: _to_string(self, maxlen)

### Function: _to_string(self, maxlen)

### Function: getter(self, fname, BField, offset, PTR)

### Function: setter(self, value, fname, BField)

### Function: getter(self, fname, BField)

### Function: setter(self, value, fname, BField)

### Function: callback()

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: getter(self, fname, BFieldPtr, offset, PTR)
