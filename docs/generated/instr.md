## AI Summary

A file named instr.py.


## Class: Compare

### Function: const_key(obj)

### Function: _pushes_back(opname)

### Function: _check_lineno(lineno)

## Class: SetLineno

## Class: Label

## Class: _Variable

## Class: CellVar

## Class: FreeVar

### Function: _check_arg_int(name, arg)

## Class: Instr

**Description:** Abstract instruction.

### Function: __init__(self, lineno)

### Function: lineno(self)

### Function: __eq__(self, other)

### Function: __init__(self, name)

### Function: __eq__(self, other)

### Function: __str__(self)

### Function: __repr__(self)

### Function: __init__(self, name, arg)

### Function: _check_arg(self, name, opcode, arg)

### Function: _set(self, name, arg, lineno)

### Function: set(self, name, arg)

**Description:** Modify the instruction in-place.

Replace name and arg attributes. Don't modify lineno.

### Function: require_arg(self)

**Description:** Does the instruction require an argument?

### Function: name(self)

### Function: name(self, name)

### Function: opcode(self)

### Function: opcode(self, op)

### Function: arg(self)

### Function: arg(self, arg)

### Function: lineno(self)

### Function: lineno(self, lineno)

### Function: stack_effect(self, jump)

### Function: pre_and_post_stack_effect(self, jump)

### Function: copy(self)

### Function: __repr__(self)

### Function: _cmp_key(self, labels)

### Function: __eq__(self, other)

### Function: _has_jump(opcode)

### Function: has_jump(self)

### Function: is_cond_jump(self)

**Description:** Is a conditional jump?

### Function: is_uncond_jump(self)

**Description:** Is an unconditional jump?

### Function: is_final(self)
