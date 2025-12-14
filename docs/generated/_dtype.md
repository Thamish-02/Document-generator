## AI Summary

A file named _dtype.py.


### Function: _kind_name(dtype)

### Function: __str__(dtype)

### Function: __repr__(dtype)

### Function: _unpack_field(dtype, offset, title)

**Description:** Helper function to normalize the items in dtype.fields.

Call as:

dtype, offset, title = _unpack_field(*dtype.fields[name])

### Function: _isunsized(dtype)

### Function: _construction_repr(dtype, include_align, short)

**Description:** Creates a string repr of the dtype, excluding the 'dtype()' part
surrounding the object. This object may be a string, a list, or
a dict depending on the nature of the dtype. This
is the object passed as the first parameter to the dtype
constructor, and if no additional constructor parameters are
given, will reproduce the exact memory layout.

Parameters
----------
short : bool
    If true, this creates a shorter repr using 'kind' and 'itemsize',
    instead of the longer type name.

include_align : bool
    If true, this includes the 'align=True' parameter
    inside the struct dtype construction dict when needed. Use this flag
    if you want a proper repr string without the 'dtype()' part around it.

    If false, this does not preserve the
    'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
    struct arrays like the regular repr does, because the 'align'
    flag is not part of first dtype constructor parameter. This
    mode is intended for a full 'repr', where the 'align=True' is
    provided as the second parameter.

### Function: _scalar_str(dtype, short)

### Function: _byte_order_str(dtype)

**Description:** Normalize byteorder to '<' or '>' 

### Function: _datetime_metadata_str(dtype)

### Function: _struct_dict_str(dtype, includealignedflag)

### Function: _aligned_offset(offset, alignment)

### Function: _is_packed(dtype)

**Description:** Checks whether the structured data type in 'dtype'
has a simple layout, where all the fields are in order,
and follow each other with no alignment padding.

When this returns true, the dtype can be reconstructed
from a list of the field names and dtypes with no additional
dtype parameters.

Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.

### Function: _struct_list_str(dtype)

### Function: _struct_str(dtype, include_align)

### Function: _subarray_str(dtype)

### Function: _name_includes_bit_suffix(dtype)

### Function: _name_get(dtype)
