## AI Summary

A file named concrete.py.


### Function: _set_docstring(code, consts)

## Class: ConcreteInstr

**Description:** Concrete instruction.

arg must be an integer in the range 0..2147483647.

It has a read-only size attribute.

## Class: ConcreteBytecode

## Class: _ConvertBytecodeToConcrete

### Function: __init__(self, name, arg)

### Function: _check_arg(self, name, opcode, arg)

### Function: _set(self, name, arg, lineno)

### Function: size(self)

### Function: _cmp_key(self, labels)

### Function: get_jump_target(self, instr_offset)

### Function: assemble(self)

### Function: disassemble(cls, lineno, code, offset)

### Function: __init__(self, instructions)

### Function: __iter__(self)

### Function: _check_instr(self, instr)

### Function: _copy_attr_from(self, bytecode)

### Function: __repr__(self)

### Function: __eq__(self, other)

### Function: from_code(code)

### Function: _normalize_lineno(instructions, first_lineno)

### Function: _assemble_code(self)

### Function: _assemble_lnotab(first_lineno, linenos)

### Function: _pack_linetable(doff, dlineno, linetable)

### Function: _assemble_linestable(self, first_lineno, linenos)

### Function: _remove_extended_args(instructions)

### Function: compute_stacksize(self)

### Function: to_code(self, stacksize)

### Function: to_bytecode(self)

### Function: __init__(self, code)

### Function: add_const(self, value)

### Function: add(names, name)

### Function: concrete_instructions(self)

### Function: compute_jumps(self)

### Function: to_concrete_bytecode(self, compute_jumps_passes)
