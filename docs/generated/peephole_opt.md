## AI Summary

A file named peephole_opt.py.


## Class: ExitUnchanged

**Description:** Exception used to skip the peephole optimizer

## Class: PeepholeOptimizer

**Description:** Python reimplementation of the peephole optimizer.

Copy of the C comment:

Perform basic peephole optimizations to components of a code object.
The consts object should still be in list form to allow new constants
to be appended.

To keep the optimizer simple, it bails out (does nothing) for code that
has a length over 32,700, and does not calculate extended arguments.
That allows us to avoid overflow and sign issues. Likewise, it bails when
the lineno table has complex encoding for gaps >= 255. EXTENDED_ARG can
appear before MAKE_FUNCTION; in this case both opcodes are skipped.
EXTENDED_ARG preceding any other opcode causes the optimizer to bail.

Optimizations are restricted to simple transformations occuring within a
single basic block.  All transformations keep the code size the same or
smaller.  For those that reduce size, the gaps are initially filled with
NOPs.  Later those NOPs are removed and the jump addresses retargeted in
a single pass.  Code offset is adjusted accordingly.

## Class: CodeTransformer

### Function: __init__(self)

### Function: check_result(self, value)

### Function: replace_load_const(self, nconst, instr, result)

### Function: eval_LOAD_CONST(self, instr)

### Function: unaryop(self, op, instr)

### Function: eval_UNARY_POSITIVE(self, instr)

### Function: eval_UNARY_NEGATIVE(self, instr)

### Function: eval_UNARY_INVERT(self, instr)

### Function: get_next_instr(self, name)

### Function: eval_UNARY_NOT(self, instr)

### Function: binop(self, op, instr)

### Function: eval_BINARY_ADD(self, instr)

### Function: eval_BINARY_SUBTRACT(self, instr)

### Function: eval_BINARY_MULTIPLY(self, instr)

### Function: eval_BINARY_TRUE_DIVIDE(self, instr)

### Function: eval_BINARY_FLOOR_DIVIDE(self, instr)

### Function: eval_BINARY_MODULO(self, instr)

### Function: eval_BINARY_POWER(self, instr)

### Function: eval_BINARY_LSHIFT(self, instr)

### Function: eval_BINARY_RSHIFT(self, instr)

### Function: eval_BINARY_AND(self, instr)

### Function: eval_BINARY_OR(self, instr)

### Function: eval_BINARY_XOR(self, instr)

### Function: eval_BINARY_SUBSCR(self, instr)

### Function: replace_container_of_consts(self, instr, container_type)

### Function: build_tuple_unpack_seq(self, instr)

### Function: build_tuple(self, instr, container_type)

### Function: eval_BUILD_TUPLE(self, instr)

### Function: eval_BUILD_LIST(self, instr)

### Function: eval_BUILD_SET(self, instr)

### Function: eval_COMPARE_OP(self, instr)

### Function: jump_if_or_pop(self, instr)

### Function: eval_JUMP_IF_FALSE_OR_POP(self, instr)

### Function: eval_JUMP_IF_TRUE_OR_POP(self, instr)

### Function: eval_NOP(self, instr)

### Function: optimize_jump_to_cond_jump(self, instr)

### Function: optimize_jump(self, instr)

### Function: iterblock(self, block)

### Function: optimize_block(self, block)

### Function: remove_dead_blocks(self)

### Function: optimize_cfg(self, cfg)

### Function: optimize(self, code_obj)

### Function: code_transformer(self, code, context)
