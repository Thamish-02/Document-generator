## AI Summary

A file named cfg.py.


## Class: BasicBlock

### Function: _compute_stack_size(block, size, maxsize)

**Description:** Generator used to reduce the use of function stacks.

This allows to avoid nested recursion and allow to treat more cases.

HOW-TO:
    Following the methods of Trampoline
    (see https://en.wikipedia.org/wiki/Trampoline_(computing)),

    We yield either:

    - the arguments that would be used in the recursive calls, i.e,
      'yield block, size, maxsize' instead of making a recursive call
      '_compute_stack_size(block, size, maxsize)', if we encounter an
      instruction jumping to another block or if the block is linked to
      another one (ie `next_block` is set)
    - the required stack from the stack if we went through all the instructions
      or encountered an unconditional jump.

    In the first case, the calling function is then responsible for creating a
    new generator with those arguments, iterating over it till exhaustion to
    determine the stacksize required by the block and resuming this function
    with the determined stacksize.

## Class: ControlFlowGraph

### Function: __init__(self, instructions)

### Function: __iter__(self)

### Function: __getitem__(self, index)

### Function: copy(self)

### Function: legalize(self, first_lineno)

**Description:** Check that all the element of the list are valid and remove SetLineno.

### Function: get_jump(self)

### Function: update_size(pre_delta, post_delta, size, maxsize)

### Function: __init__(self)

### Function: legalize(self)

**Description:** Legalize all blocks.

### Function: get_block_index(self, block)

### Function: _add_block(self, block)

### Function: add_block(self, instructions)

### Function: compute_stacksize(self)

**Description:** Compute the stack size by iterating through the blocks

The implementation make use of a generator function to avoid issue with
deeply nested recursions.

### Function: __repr__(self)

### Function: get_instructions(self)

### Function: __eq__(self, other)

### Function: __len__(self)

### Function: __iter__(self)

### Function: __getitem__(self, index)

### Function: __delitem__(self, index)

### Function: split_block(self, block, index)

### Function: from_bytecode(bytecode)

### Function: to_bytecode(self)

**Description:** Convert to Bytecode.

### Function: to_code(self, stacksize)

**Description:** Convert to code.
