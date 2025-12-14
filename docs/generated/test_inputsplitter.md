## AI Summary

A file named test_inputsplitter.py.


### Function: mini_interactive_loop(input_func)

**Description:** Minimal example of the logic of an interactive interpreter loop.

This serves as an example, and it is used by the test system with a fake
raw_input that simulates interactive input.

### Function: pseudo_input(lines)

**Description:** Return a function that acts like raw_input but feeds the input list.

### Function: test_spaces()

### Function: test_remove_comments()

### Function: test_get_input_encoding()

## Class: NoInputEncodingTestCase

## Class: InputSplitterTestCase

## Class: InteractiveLoopTestCase

**Description:** Tests for an interactive loop like a python shell.
    

## Class: IPythonInputTestCase

**Description:** By just creating a new class whose .isp is a different instance, we
re-run the same test battery on the new input splitter.

In addition, this runs the tests over the syntax and syntax_ml dicts that
were tested by individual functions, as part of the OO interface.

It also makes some checks on the raw buffer storage.

### Function: test_last_blank()

### Function: test_last_two_blanks()

## Class: CellMagicsCommon

## Class: CellModeCellMagics

## Class: LineModeCellMagics

### Function: test_find_next_indent()

### Function: raw_in(prompt)

### Function: setUp(self)

### Function: test(self)

### Function: tearDown(self)

### Function: setUp(self)

### Function: test_reset(self)

### Function: test_source(self)

### Function: test_indent(self)

### Function: test_indent2(self)

### Function: test_indent3(self)

### Function: test_indent4(self)

### Function: test_dedent_pass(self)

### Function: test_dedent_break(self)

### Function: test_dedent_continue(self)

### Function: test_dedent_raise(self)

### Function: test_dedent_return(self)

### Function: test_push(self)

### Function: test_push2(self)

### Function: test_push3(self)

### Function: test_push_accepts_more(self)

### Function: test_push_accepts_more2(self)

### Function: test_push_accepts_more3(self)

### Function: test_push_accepts_more4(self)

### Function: test_push_accepts_more5(self)

### Function: test_continuation(self)

### Function: test_syntax_error(self)

### Function: test_unicode(self)

### Function: test_line_continuation(self)

**Description:** Test issue #2108.

### Function: test_check_complete(self)

### Function: check_ns(self, lines, ns)

**Description:** Validate that the given input lines produce the resulting namespace.

Note: the input lines are given exactly as they would be typed in an
auto-indenting environment, as mini_interactive_loop above already does
auto-indenting and prepends spaces to the input.

### Function: test_simple(self)

### Function: test_simple2(self)

### Function: test_xy(self)

### Function: test_abc(self)

### Function: test_multi(self)

### Function: setUp(self)

### Function: test_syntax(self)

**Description:** Call all single-line syntax tests from the main object

### Function: test_syntax_multiline(self)

### Function: test_syntax_multiline_cell(self)

### Function: test_cellmagic_preempt(self)

### Function: test_multiline_passthrough(self)

### Function: test_whole_cell(self)

### Function: test_cellmagic_help(self)

### Function: tearDown(self)

### Function: test_incremental(self)

### Function: test_no_strip_coding(self)

### Function: test_incremental(self)

## Class: X

## Class: CommentTransformer

### Function: __init__(self)

### Function: push(self, line)

### Function: reset(self)
