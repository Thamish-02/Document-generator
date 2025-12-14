## AI Summary

A file named context_i386.py.


## Class: FLOATING_SAVE_AREA

## Class: CONTEXT

## Class: Context

**Description:** Register context dictionary for the i386 architecture.

## Class: _LDT_ENTRY_BYTES_

## Class: _LDT_ENTRY_BITS_

## Class: _LDT_ENTRY_HIGHWORD_

## Class: LDT_ENTRY

### Function: GetThreadSelectorEntry(hThread, dwSelector)

### Function: GetThreadContext(hThread, ContextFlags, raw)

### Function: SetThreadContext(hThread, lpContext)

### Function: from_dict(cls, fsa)

**Description:** Instance a new structure from a Python dictionary.

### Function: to_dict(self)

**Description:** Convert a structure into a Python dictionary.

### Function: from_dict(cls, ctx)

**Description:** Instance a new structure from a Python dictionary.

### Function: to_dict(self)

**Description:** Convert a structure into a Python native type.

### Function: __get_pc(self)

### Function: __set_pc(self, value)

### Function: __get_sp(self)

### Function: __set_sp(self, value)

### Function: __get_fp(self)

### Function: __set_fp(self, value)
