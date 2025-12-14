## AI Summary

A file named compiler.py.


### Function: optimizeconst(f)

### Function: _make_binop(op)

### Function: _make_unop(op)

### Function: generate(node, environment, name, filename, stream, defer_init, optimized)

**Description:** Generate the python source for a node tree.

### Function: has_safe_repr(value)

**Description:** Does the node have a safe representation?

### Function: find_undeclared(nodes, names)

**Description:** Check if the names passed are accessed undeclared.  The return value
is a set of all the undeclared names from the sequence of names found.

## Class: MacroRef

## Class: Frame

**Description:** Holds compile time information for us.

## Class: VisitorExit

**Description:** Exception used by the `UndeclaredNameVisitor` to signal a stop.

## Class: DependencyFinderVisitor

**Description:** A visitor that collects filter and test calls.

## Class: UndeclaredNameVisitor

**Description:** A visitor that checks if a name is accessed without being
declared.  This is different from the frame visitor as it will
not stop at closure frames.

## Class: CompilerExit

**Description:** Raised if the compiler encountered a situation where it just
doesn't make sense to further process the code.  Any block that
raises such an exception is not further processed.

## Class: CodeGenerator

### Function: new_func(self, node, frame)

### Function: visitor(self, node, frame)

### Function: visitor(self, node, frame)

### Function: __init__(self, node)

### Function: __init__(self, eval_ctx, parent, level)

### Function: copy(self)

**Description:** Create a copy of the current one.

### Function: inner(self, isolated)

**Description:** Return an inner frame.

### Function: soft(self)

**Description:** Return a soft frame.  A soft frame may not be modified as
standalone thing as it shares the resources with the frame it
was created of, but it's not a rootlevel frame any longer.

This is only used to implement if-statements and conditional
expressions.

### Function: __init__(self)

### Function: visit_Filter(self, node)

### Function: visit_Test(self, node)

### Function: visit_Block(self, node)

**Description:** Stop visiting at blocks.

### Function: __init__(self, names)

### Function: visit_Name(self, node)

### Function: visit_Block(self, node)

**Description:** Stop visiting a blocks.

### Function: __init__(self, environment, name, filename, stream, defer_init, optimized)

### Function: optimized(self)

### Function: fail(self, msg, lineno)

**Description:** Fail with a :exc:`TemplateAssertionError`.

### Function: temporary_identifier(self)

**Description:** Get a new unique identifier.

### Function: buffer(self, frame)

**Description:** Enable buffering for the frame from that point onwards.

### Function: return_buffer_contents(self, frame, force_unescaped)

**Description:** Return the buffer contents of the frame.

### Function: indent(self)

**Description:** Indent by one.

### Function: outdent(self, step)

**Description:** Outdent by step.

### Function: start_write(self, frame, node)

**Description:** Yield or write into the frame buffer.

### Function: end_write(self, frame)

**Description:** End the writing process started by `start_write`.

### Function: simple_write(self, s, frame, node)

**Description:** Simple shortcut for start_write + write + end_write.

### Function: blockvisit(self, nodes, frame)

**Description:** Visit a list of nodes as block in a frame.  If the current frame
is no buffer a dummy ``if 0: yield None`` is written automatically.

### Function: write(self, x)

**Description:** Write a string into the output stream.

### Function: writeline(self, x, node, extra)

**Description:** Combination of newline and write.

### Function: newline(self, node, extra)

**Description:** Add one or more newlines before the next write.

### Function: signature(self, node, frame, extra_kwargs)

**Description:** Writes a function call to the stream for the current node.
A leading comma is added automatically.  The extra keyword
arguments may not include python keywords otherwise a syntax
error could occur.  The extra keyword arguments should be given
as python dict.

### Function: pull_dependencies(self, nodes)

**Description:** Find all filter and test names used in the template and
assign them to variables in the compiled namespace. Checking
that the names are registered with the environment is done when
compiling the Filter and Test nodes. If the node is in an If or
CondExpr node, the check is done at runtime instead.

.. versionchanged:: 3.0
    Filters and tests in If and CondExpr nodes are checked at
    runtime instead of compile time.

### Function: enter_frame(self, frame)

### Function: leave_frame(self, frame, with_python_scope)

### Function: choose_async(self, async_value, sync_value)

### Function: func(self, name)

### Function: macro_body(self, node, frame)

**Description:** Dump the function def of a macro or call block.

### Function: macro_def(self, macro_ref, frame)

**Description:** Dump the macro definition for the def created by macro_body.

### Function: position(self, node)

**Description:** Return a human readable position for the node.

### Function: dump_local_context(self, frame)

### Function: write_commons(self)

**Description:** Writes a common preamble that is used by root and block functions.
Primarily this sets up common local helpers and enforces a generator
through a dead branch.

### Function: push_parameter_definitions(self, frame)

**Description:** Pushes all parameter targets from the given frame into a local
stack that permits tracking of yet to be assigned parameters.  In
particular this enables the optimization from `visit_Name` to skip
undefined expressions for parameters in macros as macros can reference
otherwise unbound parameters.

### Function: pop_parameter_definitions(self)

**Description:** Pops the current parameter definitions set.

### Function: mark_parameter_stored(self, target)

**Description:** Marks a parameter in the current parameter definitions as stored.
This will skip the enforced undefined checks.

### Function: push_context_reference(self, target)

### Function: pop_context_reference(self)

### Function: get_context_ref(self)

### Function: get_resolve_func(self)

### Function: derive_context(self, frame)

### Function: parameter_is_undeclared(self, target)

**Description:** Checks if a given target is an undeclared parameter.

### Function: push_assign_tracking(self)

**Description:** Pushes a new layer for assignment tracking.

### Function: pop_assign_tracking(self, frame)

**Description:** Pops the topmost level for assignment tracking and updates the
context variables if necessary.

### Function: visit_Template(self, node, frame)

### Function: visit_Block(self, node, frame)

**Description:** Call a block and register it for the template.

### Function: visit_Extends(self, node, frame)

**Description:** Calls the extender.

### Function: visit_Include(self, node, frame)

**Description:** Handles includes.

### Function: _import_common(self, node, frame)

### Function: visit_Import(self, node, frame)

**Description:** Visit regular imports.

### Function: visit_FromImport(self, node, frame)

**Description:** Visit named imports.

### Function: visit_For(self, node, frame)

### Function: visit_If(self, node, frame)

### Function: visit_Macro(self, node, frame)

### Function: visit_CallBlock(self, node, frame)

### Function: visit_FilterBlock(self, node, frame)

### Function: visit_With(self, node, frame)

### Function: visit_ExprStmt(self, node, frame)

## Class: _FinalizeInfo

### Function: _default_finalize(value)

**Description:** The default finalize function if the environment isn't
configured with one. Or, if the environment has one, this is
called on that function's output for constants.

### Function: _make_finalize(self)

**Description:** Build the finalize function to be used on constants and at
runtime. Cached so it's only created once for all output nodes.

Returns a ``namedtuple`` with the following attributes:

``const``
    A function to finalize constant data at compile time.

``src``
    Source code to output around nodes to be evaluated at
    runtime.

### Function: _output_const_repr(self, group)

**Description:** Given a group of constant values converted from ``Output``
child nodes, produce a string to write to the template module
source.

### Function: _output_child_to_const(self, node, frame, finalize)

**Description:** Try to optimize a child of an ``Output`` node by trying to
convert it to constant, finalized data at compile time.

If :exc:`Impossible` is raised, the node is not constant and
will be evaluated at runtime. Any other exception will also be
evaluated at runtime for easier debugging.

### Function: _output_child_pre(self, node, frame, finalize)

**Description:** Output extra source code before visiting a child of an
``Output`` node.

### Function: _output_child_post(self, node, frame, finalize)

**Description:** Output extra source code after visiting a child of an
``Output`` node.

### Function: visit_Output(self, node, frame)

### Function: visit_Assign(self, node, frame)

### Function: visit_AssignBlock(self, node, frame)

### Function: visit_Name(self, node, frame)

### Function: visit_NSRef(self, node, frame)

### Function: visit_Const(self, node, frame)

### Function: visit_TemplateData(self, node, frame)

### Function: visit_Tuple(self, node, frame)

### Function: visit_List(self, node, frame)

### Function: visit_Dict(self, node, frame)

### Function: visit_Concat(self, node, frame)

### Function: visit_Compare(self, node, frame)

### Function: visit_Operand(self, node, frame)

### Function: visit_Getattr(self, node, frame)

### Function: visit_Getitem(self, node, frame)

### Function: visit_Slice(self, node, frame)

### Function: _filter_test_common(self, node, frame, is_filter)

### Function: visit_Filter(self, node, frame)

### Function: visit_Test(self, node, frame)

### Function: visit_CondExpr(self, node, frame)

### Function: visit_Call(self, node, frame, forward_caller)

### Function: visit_Keyword(self, node, frame)

### Function: visit_MarkSafe(self, node, frame)

### Function: visit_MarkSafeIfAutoescape(self, node, frame)

### Function: visit_EnvironmentAttribute(self, node, frame)

### Function: visit_ExtensionAttribute(self, node, frame)

### Function: visit_ImportedName(self, node, frame)

### Function: visit_InternalName(self, node, frame)

### Function: visit_ContextReference(self, node, frame)

### Function: visit_DerivedContextReference(self, node, frame)

### Function: visit_Continue(self, node, frame)

### Function: visit_Break(self, node, frame)

### Function: visit_Scope(self, node, frame)

### Function: visit_OverlayScope(self, node, frame)

### Function: visit_EvalContextModifier(self, node, frame)

### Function: visit_ScopedEvalContextModifier(self, node, frame)

### Function: loop_body()

### Function: write_expr2()

### Function: finalize(value)

### Function: finalize(value)
