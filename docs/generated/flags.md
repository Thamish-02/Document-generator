## AI Summary

A file named flags.py.


## Class: CompilerFlags

**Description:** Possible values of the co_flags attribute of Code object.

Note: We do not rely on inspect values here as some of them are missing and
furthermore would be version dependent.

### Function: infer_flags(bytecode, is_async)

**Description:** Infer the proper flags for a bytecode based on the instructions.

Because the bytecode does not have enough context to guess if a function
is asynchronous the algorithm tries to be conservative and will never turn
a previously async code into a sync one.

Parameters
----------
bytecode : Bytecode | ConcreteBytecode | ControlFlowGraph
    Bytecode for which to infer the proper flags
is_async : bool | None, optional
    Force the code to be marked as asynchronous if True, prevent it from
    being marked as asynchronous if False and simply infer the best
    solution based on the opcode and the existing flag if None.
