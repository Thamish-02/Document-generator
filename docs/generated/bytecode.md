## AI Summary

A file named bytecode.py.


## Class: BaseBytecode

## Class: _BaseBytecodeList

**Description:** List subclass providing type stable slicing and copying.

## Class: _InstrList

## Class: Bytecode

### Function: __init__(self)

### Function: _copy_attr_from(self, bytecode)

### Function: __eq__(self, other)

### Function: flags(self)

### Function: flags(self, value)

### Function: update_flags(self)

### Function: __getitem__(self, index)

### Function: copy(self)

### Function: legalize(self)

**Description:** Check that all the element of the list are valid and remove SetLineno.

### Function: __iter__(self)

### Function: _check_instr(self, instr)

### Function: _flat(self)

### Function: __eq__(self, other)

### Function: __init__(self, instructions)

### Function: __iter__(self)

### Function: _check_instr(self, instr)

### Function: _copy_attr_from(self, bytecode)

### Function: from_code(code)

### Function: compute_stacksize(self)

### Function: to_code(self, compute_jumps_passes, stacksize)

### Function: to_concrete_bytecode(self, compute_jumps_passes)
