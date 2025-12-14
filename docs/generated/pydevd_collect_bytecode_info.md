## AI Summary

A file named pydevd_collect_bytecode_info.py.


## Class: TryExceptInfo

## Class: ReturnInfo

### Function: _get_line(op_offset_to_line, op_offset, firstlineno, search)

### Function: debug(s)

### Function: iter_instructions(co)

### Function: collect_return_info(co, use_func_first_line)

## Class: _Visitor

### Function: collect_try_except_info_from_source(filename)

### Function: collect_try_except_info_from_contents(contents, filename)

## Class: _MsgPart

## Class: _Disassembler

### Function: code_to_bytecode_representation(co, use_func_first_line)

**Description:** A simple disassemble of bytecode.

It does not attempt to provide the full Python source code, rather, it provides a low-level
representation of the bytecode, respecting the lines (so, its target is making the bytecode
easier to grasp and not providing the original source code).

Note that it does show jump locations/targets and converts some common bytecode constructs to
Python code to make it a bit easier to understand.

### Function: __init__(self, try_line, ignore)

**Description:** :param try_line:
:param ignore:
    Usually we should ignore any block that's not a try..except
    (this can happen for finally blocks, with statements, etc, for
    which we create temporary entries).

### Function: is_line_in_try_block(self, line)

### Function: is_line_in_except_block(self, line)

### Function: __str__(self)

### Function: __init__(self, return_line)

### Function: __str__(self)

## Class: _TargetInfo

### Function: _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx)

### Function: collect_try_except_info(co, use_func_first_line)

### Function: __init__(self)

### Function: generic_visit(self, node)

### Function: visit_Try(self, node)

### Function: visit_ExceptHandler(self, node)

### Function: __init__(self, line, tok)

### Function: __str__(self)

### Function: add_to_line_to_contents(cls, obj, line_to_contents, line)

### Function: __init__(self, co, firstlineno, level)

### Function: min_line(self)

### Function: max_line(self)

### Function: _lookahead(self)

**Description:** This handles and converts some common constructs from bytecode to actual source code.

It may change the list of instructions.

### Function: _decorate_jump_target(self, instruction, instruction_repr)

### Function: _create_msg_part(self, instruction, tok, line)

### Function: _next_instruction_to_str(self, line_to_contents)

### Function: build_line_to_contents(self)

### Function: disassemble(self)

### Function: __init__(self, except_end_instruction, jump_if_not_exc_instruction)

### Function: __str__(self)

## Class: _TargetInfo

### Function: _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx)

### Function: collect_try_except_info(co, use_func_first_line)

### Function: visit_Raise(self, node)

### Function: visit_Raise(self, node)

### Function: __init__(self, except_end_instruction, jump_if_not_exc_instruction)

### Function: __str__(self)

### Function: collect_try_except_info(co, use_func_first_line)

**Description:** Note: if the filename is available and we can get the source,
`collect_try_except_info_from_source` is preferred (this is kept as
a fallback for cases where sources aren't available).
