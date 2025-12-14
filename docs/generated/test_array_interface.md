## AI Summary

A file named test_array_interface.py.


### Function: get_module(tmp_path)

**Description:** Some codes to generate data and manage temporary buffers use when
sharing with numpy via the array interface protocol.

### Function: test_cstruct(get_module)

## Class: data_source

**Description:** This class is for testing the timing of the PyCapsule destructor
invoked when numpy release its reference to the shared data as part of
the numpy array interface protocol. If the PyCapsule destructor is
called early the shared data is freed and invalid memory accesses will
occur.

### Function: __init__(self, size, value)

### Function: __array_struct__(self)
