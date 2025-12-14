## AI Summary

A file named context_amd64.py.


## Class: XMM_SAVE_AREA32

## Class: _CONTEXT_FLTSAVE_STRUCT

## Class: _CONTEXT_FLTSAVE_UNION

## Class: CONTEXT

## Class: Context

**Description:** Register context dictionary for the amd64 architecture.

## Class: _LDT_ENTRY_BYTES_

## Class: _LDT_ENTRY_BITS_

## Class: _LDT_ENTRY_HIGHWORD_

## Class: LDT_ENTRY

## Class: WOW64_FLOATING_SAVE_AREA

## Class: WOW64_CONTEXT

## Class: WOW64_LDT_ENTRY

### Function: GetThreadSelectorEntry(hThread, dwSelector)

### Function: GetThreadContext(hThread, ContextFlags, raw)

### Function: SetThreadContext(hThread, lpContext)

### Function: Wow64GetThreadSelectorEntry(hThread, dwSelector)

### Function: Wow64ResumeThread(hThread)

### Function: Wow64SuspendThread(hThread)

### Function: Wow64GetThreadContext(hThread, ContextFlags)

### Function: Wow64SetThreadContext(hThread, lpContext)

### Function: from_dict(self)

### Function: to_dict(self)

### Function: from_dict(self)

### Function: to_dict(self)

### Function: from_dict(self)

### Function: to_dict(self)

### Function: from_dict(cls, ctx)

**Description:** Instance a new structure from a Python native type.

### Function: to_dict(self)

**Description:** Convert a structure into a Python dictionary.

### Function: __get_pc(self)

### Function: __set_pc(self, value)

### Function: __get_sp(self)

### Function: __set_sp(self, value)

### Function: __get_fp(self)

### Function: __set_fp(self, value)
