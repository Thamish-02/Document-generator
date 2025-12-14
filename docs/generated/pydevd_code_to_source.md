## AI Summary

A file named pydevd_code_to_source.py.


## Class: _Stack

## Class: _Token

## Class: _Writer

## Class: _BaseHandler

### Function: _register(cls)

## Class: _BasePushHandler

## Class: _BaseLoadHandler

## Class: _LoadBuildClass

## Class: _LoadConst

## Class: _LoadName

## Class: _LoadGlobal

## Class: _LoadFast

## Class: _GetIter

**Description:** Implements TOS = iter(TOS).

## Class: _ForIter

**Description:** TOS is an iterator. Call its __next__() method. If this yields a new value, push it on the stack
(leaving the iterator below it). If the iterator indicates it is exhausted TOS is popped, and
the byte code counter is incremented by delta.

## Class: _StoreName

**Description:** Implements name = TOS. namei is the index of name in the attribute co_names of the code object.
The compiler tries to use STORE_FAST or STORE_GLOBAL if possible.

## Class: _ReturnValue

**Description:** Returns with TOS to the caller of the function.

## Class: _CallFunction

**Description:** CALL_FUNCTION(argc)

    Calls a callable object with positional arguments. argc indicates the number of positional
    arguments. The top of the stack contains positional arguments, with the right-most argument
    on top. Below the arguments is a callable object to call. CALL_FUNCTION pops all arguments
    and the callable object off the stack, calls the callable object with those arguments, and
    pushes the return value returned by the callable object.

    Changed in version 3.6: This opcode is used only for calls with positional arguments.

## Class: _MakeFunctionPy3

**Description:** Pushes a new function object on the stack. From bottom to top, the consumed stack must consist
of values if the argument carries a specified flag value

    0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order

    0x02 a dictionary of keyword-only parameters' default values

    0x04 an annotation dictionary

    0x08 a tuple containing cells for free variables, making a closure

    the code associated with the function (at TOS1)

    the qualified name of the function (at TOS)

### Function: _print_after_info(line_contents, stream)

### Function: _compose_line_contents(line_contents, previous_line_tokens)

## Class: _PyCodeToSource

### Function: code_obj_to_source(co)

**Description:** Converts a code object to source code to provide a suitable representation for the compiler when
the actual source code is not found.

This is a work in progress / proof of concept / not ready to be used.

### Function: __init__(self)

### Function: push(self, obj)

### Function: pop(self)

### Function: __init__(self, i_line, instruction, tok, priority, after, end_of_line)

**Description:** :param i_line:
:param instruction:
:param tok:
:param priority:
:param after:
:param end_of_line:
    Marker to signal only after all the other tokens have been written.

### Function: mark_after(self, v)

### Function: get_after_tokens(self)

### Function: __repr__(self)

### Function: __init__(self)

### Function: get_line(self, line)

### Function: indent(self, line)

### Function: dedent(self, line)

### Function: write(self, line, token)

### Function: __init__(self, i_line, instruction, stack, writer, disassembler)

### Function: _write_tokens(self)

### Function: _handle(self)

### Function: __repr__(self)

### Function: _handle(self)

### Function: _handle(self)

### Function: _handle(self)

### Function: _handle(self)

### Function: store_in_name(self, store_name)

### Function: _handle(self)

### Function: _handle(self)

### Function: _handle(self)

### Function: _handle(self)

### Function: __init__(self, co, memo)

### Function: _process_next(self, i_line)

### Function: build_line_to_contents(self)

### Function: merge_code(self, code)

### Function: disassemble(self)
