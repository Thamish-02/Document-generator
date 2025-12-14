## AI Summary

A file named _position_node_finder.py.


### Function: parents(node)

### Function: node_and_parents(node)

### Function: get_instructions(code)

## Class: PositionNodeFinder

**Description:** Mapping bytecode to ast-node based on the source positions, which where introduced in pyhon 3.11.
In general every ast-node can be exactly referenced by its begin/end line/col_offset, which is stored in the bytecode.
There are only some exceptions for methods and attributes.

### Function: __init__(self, frame, stmts, tree, lasti, source)

### Function: test_for_decorator(self, node, index)

### Function: fix_result(self, node, instruction)

### Function: known_issues(self, node, instruction)

### Function: annotation_header_size(self)

### Function: is_except_cleanup(inst, node)

### Function: verify(self, node, instruction)

**Description:** checks if this node could gererate this instruction

### Function: instruction(self, index)

### Function: instruction_before(self, instruction)

### Function: opname(self, index)

### Function: find_node(self, index, match_positions, typ)

### Function: inst_match(opnames)

**Description:** match instruction

Parameters:
    opnames: (str|Seq[str]): inst.opname has to be equal to or in `opname`
    **kwargs: every arg has to match inst.arg

Returns:
    True if all conditions match the instruction

### Function: node_match(node_type)

**Description:** match the ast-node

Parameters:
    node_type: type of the node
    **kwargs: every `arg` has to be equal `node.arg`
            or `node.arg` has to be an instance of `arg` if it is a type.
