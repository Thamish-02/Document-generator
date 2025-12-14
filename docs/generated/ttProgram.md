## AI Summary

A file named ttProgram.py.


### Function: bitRepr(value, bits)

### Function: _makeDict(instructionList)

## Class: tt_instructions_error

### Function: _skipWhite(data, pos)

## Class: Program

### Function: _test()

**Description:** >>> _test()
True

### Function: __init__(self, error)

### Function: __str__(self)

### Function: __init__(self)

### Function: fromBytecode(self, bytecode)

### Function: fromAssembly(self, assembly)

### Function: getBytecode(self)

### Function: getAssembly(self, preserve)

### Function: toXML(self, writer, ttFont)

### Function: fromXML(self, name, attrs, content, ttFont)

### Function: _assemble(self)

### Function: _disassemble(self, preserve)

### Function: __bool__(self)

**Description:** >>> p = Program()
>>> bool(p)
False
>>> bc = array.array("B", [0])
>>> p.fromBytecode(bc)
>>> bool(p)
True
>>> p.bytecode.pop()
0
>>> bool(p)
False

>>> p = Program()
>>> asm = ['SVTCA[0]']
>>> p.fromAssembly(asm)
>>> bool(p)
True
>>> p.assembly.pop()
'SVTCA[0]'
>>> bool(p)
False

### Function: __eq__(self, other)

### Function: __ne__(self, other)
