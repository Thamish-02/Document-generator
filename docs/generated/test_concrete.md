## AI Summary

A file named test_concrete.py.


## Class: ConcreteInstrTests

## Class: ConcreteBytecodeTests

## Class: ConcreteFromCodeTests

## Class: BytecodeToConcreteTests

### Function: test_constructor(self)

### Function: test_attr(self)

### Function: test_set(self)

### Function: test_set_attr(self)

### Function: test_size(self)

### Function: test_disassemble(self)

### Function: test_assemble(self)

### Function: test_get_jump_target(self)

### Function: test_repr(self)

### Function: test_eq(self)

### Function: test_attr(self)

### Function: test_invalid_types(self)

### Function: test_to_code_lnotab(self)

### Function: test_negative_lnotab(self)

### Function: test_extended_lnotab(self)

### Function: test_extended_lnotab2(self)

### Function: test_to_bytecode_consts(self)

### Function: test_cellvar(self)

### Function: test_freevar(self)

### Function: test_cellvar_freevar(self)

### Function: test_load_classderef(self)

### Function: test_explicit_stacksize(self)

### Function: test_legalize(self)

### Function: test_slice(self)

### Function: test_copy(self)

### Function: test_extended_arg(self)

### Function: test_extended_arg_make_function(self)

### Function: test_extended_arg_unpack_ex(self)

### Function: test_expected_arg_with_many_consts(self)

### Function: test_label(self)

### Function: test_label2(self)

### Function: test_label3(self)

**Description:** CPython generates useless EXTENDED_ARG 0 in some cases. We need to
properly track them as otherwise we can end up with broken offset for
jumps.

### Function: test_setlineno(self)

### Function: test_extended_jump(self)

### Function: test_jumps(self)

### Function: test_dont_merge_constants(self)

### Function: test_cellvars(self)

### Function: test_compute_jumps_convergence(self)

### Function: test_extreme_compute_jumps_convergence(self)

**Description:** Test of compute_jumps() requiring absurd number of passes.

NOTE:  This test also serves to demonstrate that there is no worst
case: the number of passes can be unlimited (or, actually, limited by
the size of the provided code).

This is an extension of test_compute_jumps_convergence.  Instead of
two jumps, where the earlier gets extended after the latter, we
instead generate a series of many jumps.  Each pass of compute_jumps()
extends one more instruction, which in turn causes the one behind it
to be extended on the next pass.

### Function: test_general_constants(self)

**Description:** Test if general object could be linked as constants.

### Function: f()

### Function: test()

### Function: test()

### Function: test_fail_extended_arg_jump(self)

## Class: BigInstr

## Class: CustomObject

## Class: UnHashableCustomObject

### Function: f()

### Function: test()

### Function: __init__(self, size)

### Function: copy(self)

### Function: assemble(self)
