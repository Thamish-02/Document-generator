## AI Summary

A file named pydevd_bytecode_utils.py.


## Class: Target

## Class: _TargetIdHashable

## Class: _StackInterpreter

**Description:** Good reference: https://github.com/python/cpython/blob/fcb55c0037baab6f98f91ee38ce84b6f874f034a/Python/ceval.c

### Function: _get_smart_step_into_targets(code)

**Description:** :return list(Target)

## Class: Variant

### Function: _convert_target_to_variant(target, start_line, end_line, call_order_cache, lasti, base)

### Function: calculate_smart_step_into_variants(frame, start_line, end_line, base)

**Description:** Calculate smart step into variants for the given line range.
:param frame:
:type frame: :py:class:`types.FrameType`
:param start_line:
:param end_line:
:return: A list of call names from the first to the last.
:note: it's guaranteed that the offsets appear in order.
:raise: :py:class:`RuntimeError` if failed to parse the bytecode or if dis cannot be used.

### Function: get_smart_step_into_variant_from_frame_offset(frame_f_lasti, variants)

**Description:** Given the frame.f_lasti, return the related `Variant`.

:note: if the offset is found before any variant available or no variants are
       available, None is returned.

:rtype: Variant|NoneType

### Function: __init__(self, arg, lineno, offset, children_targets, endlineno, startcol, endcol)

### Function: __repr__(self)

### Function: __init__(self, target)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __hash__(self)

### Function: __init__(self, bytecode)

### Function: __str__(self)

### Function: _getname(self, instr)

### Function: _getcallname(self, instr)

### Function: _no_stack_change(self, instr)

### Function: on_LOAD_GLOBAL(self, instr)

### Function: on_POP_TOP(self, instr)

### Function: on_LOAD_ATTR(self, instr)

### Function: on_LOAD_CONST(self, instr)

### Function: on_STORE_FAST(self, instr)

### Function: on_NOP(self, instr)

### Function: _handle_call_from_instr(self, func_name_instr, func_call_instr)

### Function: on_COMPARE_OP(self, instr)

### Function: on_IS_OP(self, instr)

### Function: on_BINARY_SUBSCR(self, instr)

### Function: on_LOAD_METHOD(self, instr)

### Function: on_MAKE_FUNCTION(self, instr)

### Function: on_LOAD_FAST(self, instr)

### Function: on_LOAD_ASSERTION_ERROR(self, instr)

### Function: on_CALL_METHOD(self, instr)

### Function: on_CALL(self, instr)

### Function: on_CALL_INTRINSIC_1(self, instr)

### Function: on_PUSH_NULL(self, instr)

### Function: on_KW_NAMES(self, instr)

### Function: on_RETURN_CONST(self, instr)

### Function: on_CALL_FUNCTION(self, instr)

### Function: on_CALL_FUNCTION_KW(self, instr)

### Function: on_CALL_FUNCTION_VAR(self, instr)

### Function: on_CALL_FUNCTION_VAR_KW(self, instr)

### Function: on_CALL_FUNCTION_EX(self, instr)

### Function: on_JUMP_IF_FALSE_OR_POP(self, instr)

### Function: on_JUMP_IF_NOT_EXC_MATCH(self, instr)

### Function: on_SWAP(self, instr)

### Function: on_ROT_TWO(self, instr)

### Function: on_ROT_THREE(self, instr)

### Function: on_ROT_FOUR(self, instr)

### Function: on_BUILD_LIST_FROM_ARG(self, instr)

### Function: on_BUILD_MAP(self, instr)

### Function: on_BUILD_CONST_KEY_MAP(self, instr)

### Function: on_RETURN_GENERATOR(self, instr)

### Function: on_MAP_ADD(self, instr)

### Function: on_UNPACK_SEQUENCE(self, instr)

### Function: on_BUILD_LIST(self, instr)

### Function: on_RAISE_VARARGS(self, instr)

### Function: on_INPLACE_ADD(self, instr)

### Function: on_DUP_TOP(self, instr)

### Function: on_DUP_TOP_TWO(self, instr)

### Function: on_BUILD_SLICE(self, instr)

### Function: on_STORE_SUBSCR(self, instr)

### Function: on_DELETE_SUBSCR(self, instr)

### Function: __init__(self, name, is_visited, line, offset, call_order, children_variants, endlineno, startcol, endcol)

### Function: __repr__(self)
