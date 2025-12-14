## AI Summary

A file named mrecords.py.


### Function: _checknames(descr, names)

**Description:** Checks that field names ``descr`` are not reserved keywords.

If this is the case, a default 'f%i' is substituted.  If the argument
`names` is not None, updates the field names to valid names.

### Function: _get_fieldmask(self)

## Class: MaskedRecords

**Description:** Attributes
----------
_data : recarray
    Underlying data, as a record array.
_mask : boolean array
    Mask of the records. A record is masked when all its fields are
    masked.
_fieldmask : boolean recarray
    Record array of booleans, setting the mask of each individual field
    of each record.
_fill_value : record
    Filling values for each field.

### Function: _mrreconstruct(subtype, baseclass, baseshape, basetype)

**Description:** Build a new MaskedArray from the information stored in a pickle.

### Function: fromarrays(arraylist, dtype, shape, formats, names, titles, aligned, byteorder, fill_value)

**Description:** Creates a mrecarray from a (flat) list of masked arrays.

Parameters
----------
arraylist : sequence
    A list of (masked) arrays. Each element of the sequence is first converted
    to a masked array if needed. If a 2D array is passed as argument, it is
    processed line by line
dtype : {None, dtype}, optional
    Data type descriptor.
shape : {None, integer}, optional
    Number of records. If None, shape is defined from the shape of the
    first array in the list.
formats : {None, sequence}, optional
    Sequence of formats for each individual field. If None, the formats will
    be autodetected by inspecting the fields and selecting the highest dtype
    possible.
names : {None, sequence}, optional
    Sequence of the names of each field.
fill_value : {None, sequence}, optional
    Sequence of data to be used as filling values.

Notes
-----
Lists of tuples should be preferred over lists of lists for faster processing.

### Function: fromrecords(reclist, dtype, shape, formats, names, titles, aligned, byteorder, fill_value, mask)

**Description:** Creates a MaskedRecords from a list of records.

Parameters
----------
reclist : sequence
    A list of records. Each element of the sequence is first converted
    to a masked array if needed. If a 2D array is passed as argument, it is
    processed line by line
dtype : {None, dtype}, optional
    Data type descriptor.
shape : {None,int}, optional
    Number of records. If None, ``shape`` is defined from the shape of the
    first array in the list.
formats : {None, sequence}, optional
    Sequence of formats for each individual field. If None, the formats will
    be autodetected by inspecting the fields and selecting the highest dtype
    possible.
names : {None, sequence}, optional
    Sequence of the names of each field.
fill_value : {None, sequence}, optional
    Sequence of data to be used as filling values.
mask : {nomask, sequence}, optional.
    External mask to apply on the data.

Notes
-----
Lists of tuples should be preferred over lists of lists for faster processing.

### Function: _guessvartypes(arr)

**Description:** Tries to guess the dtypes of the str_ ndarray `arr`.

Guesses by testing element-wise conversion. Returns a list of dtypes.
The array is first converted to ndarray. If the array is 2D, the test
is performed on the first line. An exception is raised if the file is
3D or more.

### Function: openfile(fname)

**Description:** Opens the file handle of file `fname`.

### Function: fromtextfile(fname, delimiter, commentchar, missingchar, varnames, vartypes)

**Description:** Creates a mrecarray from data stored in the file `filename`.

Parameters
----------
fname : {file name/handle}
    Handle of an opened file.
delimiter : {None, string}, optional
    Alphanumeric character used to separate columns in the file.
    If None, any (group of) white spacestring(s) will be used.
commentchar : {'#', string}, optional
    Alphanumeric character used to mark the start of a comment.
missingchar : {'', string}, optional
    String indicating missing data, and used to create the masks.
varnames : {None, sequence}, optional
    Sequence of the variable names. If None, a list will be created from
    the first non empty line of the file.
vartypes : {None, sequence}, optional
    Sequence of the variables dtypes. If None, it will be estimated from
    the first non-commented line.


Ultra simple: the varnames are in the header, one line

### Function: addfield(mrecord, newfield, newfieldname)

**Description:** Adds a new field to the masked record array

Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
is None, the new field name is set to 'fi', where `i` is the number of
existing fields.

### Function: __new__(cls, shape, dtype, buf, offset, strides, formats, names, titles, byteorder, aligned, mask, hard_mask, fill_value, keep_mask, copy)

### Function: __array_finalize__(self, obj)

### Function: _data(self)

**Description:** Returns the data as a recarray.

### Function: _fieldmask(self)

**Description:** Alias to mask.

### Function: __len__(self)

**Description:** Returns the length

### Function: __getattribute__(self, attr)

### Function: __setattr__(self, attr, val)

**Description:** Sets the attribute attr to the value val.

### Function: __getitem__(self, indx)

**Description:** Returns all the fields sharing the same fieldname base.

The fieldname base is either `_data` or `_mask`.

### Function: __setitem__(self, indx, value)

**Description:** Sets the given record to value.

### Function: __str__(self)

**Description:** Calculates the string representation.

### Function: __repr__(self)

**Description:** Calculates the repr representation.

### Function: view(self, dtype, type)

**Description:** Returns a view of the mrecarray.

### Function: harden_mask(self)

**Description:** Forces the mask to hard.

### Function: soften_mask(self)

**Description:** Forces the mask to soft

### Function: copy(self)

**Description:** Returns a copy of the masked record.

### Function: tolist(self, fill_value)

**Description:** Return the data portion of the array as a list.

Data items are converted to the nearest compatible Python type.
Masked values are converted to fill_value. If fill_value is None,
the corresponding entries in the output list will be ``None``.

### Function: __getstate__(self)

**Description:** Return the internal state of the masked array.

This is for pickling.

### Function: __setstate__(self, state)

**Description:** Restore the internal state of the masked array.

This is for pickling.  ``state`` is typically the output of the
``__getstate__`` output, and is a 5-tuple:

- class name
- a tuple giving the shape of the data
- a typecode for the data
- a binary string for the data
- a binary string for the mask.

### Function: __reduce__(self)

**Description:** Return a 3-tuple for pickling a MaskedArray.
