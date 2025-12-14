## AI Summary

A file named executing.py.


## Class: NotOneValueFound

### Function: only(it)

## Class: Source

**Description:** The source code of a single file and associated metadata.

The main method of interest is the classmethod `executing(frame)`.

If you want an instance of this class, don't construct it.
Ideally use the classmethod `for_frame(frame)`.
If you don't have a frame, use `for_filename(filename [, module_globals])`.
These methods cache instances by filename, so at most one instance exists per filename.

Attributes:
    - filename
    - text
    - lines
    - tree: AST parsed from text, or None if text is not valid Python
        All nodes in the tree have an extra `parent` attribute

Other methods of interest:
    - statements_at_line
    - asttokens
    - code_qualname

## Class: Executing

**Description:** Information about the operation a frame is currently executing.

Generally you will just want `node`, which is the AST node being executed,
or None if it's unknown.

If a decorator is currently being called, then:
    - `node` is a function or class definition
    - `decorator` is the expression in `node.decorator_list` being called
    - `statements == {node}`

## Class: QualnameVisitor

### Function: compile_similar_to(source, matching_code)

### Function: is_rewritten_by_pytest(code)

## Class: SentinelNodeFinder

### Function: non_sentinel_instructions(instructions, start)

**Description:** Yields (index, instruction) pairs excluding the basic
instructions introduced by the sentinel transformation

### Function: walk_both_instructions(original_instructions, original_start, instructions, start)

**Description:** Yields matching indices and instructions from the new and original instructions,
leaving out changes made by the sentinel transformation.

### Function: handle_jumps(instructions, original_instructions)

**Description:** Transforms instructions in place until it looks more like original_instructions.
This is only needed in 3.10+ where optimisations lead to more drastic changes
after the sentinel transformation.
Replaces JUMP instructions that aren't also present in original_instructions
with the sections that they jump to until a raise or return.
In some other cases duplication found in `original_instructions`
is replicated in `instructions`.

### Function: find_new_matching(orig_section, instructions)

**Description:** Yields sections of `instructions` which match `orig_section`.
The yielded sections include sentinel instructions, but these
are ignored when checking for matches.

### Function: handle_jump(original_instructions, original_start, instructions, start)

**Description:** Returns the section of instructions starting at `start` and ending
with a RETURN_VALUE or RAISE_VARARGS instruction.
There should be a matching section in original_instructions starting at original_start.
If that section doesn't appear elsewhere in original_instructions,
then also delete the returned section of instructions.

### Function: check_duplicates(original_i, orig_section, original_instructions)

**Description:** Returns True if a section of original_instructions starting somewhere other
than original_i and matching orig_section is found, i.e. orig_section is duplicated.

### Function: sections_match(orig_section, dup_section)

**Description:** Returns True if the given lists of instructions have matching linenos and opnames.

### Function: opnames_match(inst1, inst2)

### Function: get_setter(node)

### Function: statement_containing_node(node)

### Function: assert_linenos(tree)

### Function: _extract_ipython_statement(stmt)

### Function: is_ipython_cell_code_name(code_name)

### Function: is_ipython_cell_filename(filename)

### Function: is_ipython_cell_code(code_obj)

### Function: find_node_ipython(frame, lasti, stmts, source)

### Function: node_linenos(node)

### Function: __init__(self, msg, values)

### Function: __init__(self, filename, lines)

**Description:** Don't call this constructor, see the class docstring.

### Function: for_frame(cls, frame, use_cache)

**Description:** Returns the `Source` object corresponding to the file the frame is executing in.

### Function: for_filename(cls, filename, module_globals, use_cache)

### Function: _for_filename_and_lines(cls, filename, lines)

### Function: lazycache(cls, frame)

### Function: executing(cls, frame_or_tb)

**Description:** Returns an `Executing` object representing the operation
currently executing in the given frame or traceback object.

### Function: _class_local(cls, name, default)

**Description:** Returns an attribute directly associated with this class
(as opposed to subclasses), setting default if necessary

### Function: statements_at_line(self, lineno)

**Description:** Returns the statement nodes overlapping the given line.

Returns at most one statement unless semicolons are present.

If the `text` attribute is not valid python, meaning
`tree` is None, returns an empty set.

Otherwise, `Source.for_frame(frame).statements_at_line(frame.f_lineno)`
should return at least one statement.

### Function: asttext(self)

**Description:** Returns an ASTText object for getting the source of specific AST nodes.

See http://asttokens.readthedocs.io/en/latest/api-index.html

### Function: asttokens(self)

**Description:** Returns an ASTTokens object for getting the source of specific AST nodes.

See http://asttokens.readthedocs.io/en/latest/api-index.html

### Function: _asttext_base(self)

### Function: decode_source(source)

### Function: detect_encoding(source)

### Function: code_qualname(self, code)

**Description:** Imitates the __qualname__ attribute of functions for code objects.
Given:

    - A function `func`
    - A frame `frame` for an execution of `func`, meaning:
        `frame.f_code is func.__code__`

`Source.for_frame(frame).code_qualname(frame.f_code)`
will be equal to `func.__qualname__`*. Works for Python 2 as well,
where of course no `__qualname__` attribute exists.

Falls back to `code.co_name` if there is no appropriate qualname.

Based on https://github.com/wbolster/qualname

(* unless `func` is a lambda
nested inside another lambda on the same line, in which case
the outer lambda's qualname will be returned for the codes
of both lambdas)

### Function: __init__(self, frame, source, node, stmts, decorator)

### Function: code_qualname(self)

### Function: text(self)

### Function: text_range(self)

### Function: __init__(self)

### Function: add_qualname(self, node, name)

### Function: visit_FunctionDef(self, node, name)

### Function: visit_Lambda(self, node)

### Function: visit_ClassDef(self, node)

### Function: __init__(self, frame, stmts, tree, lasti, source)

### Function: find_decorator(self, stmts)

### Function: clean_instructions(self, code)

### Function: get_original_clean_instructions(self)

### Function: matching_nodes(self, exprs)

### Function: compile_instructions(self)

### Function: find_codes(self, root_code)

### Function: get_actual_current_instruction(self, lasti)

**Description:** Get the instruction corresponding to the current
frame offset, skipping EXTENDED_ARG instructions

### Function: get_lines()

### Function: matches(c)

### Function: finder(code)

### Function: setter(new_node)

### Function: setter(new_node)
