## AI Summary

A file named nodes.py.


## Class: Impossible

**Description:** Raised if the node could not perform a requested action.

## Class: NodeType

**Description:** A metaclass for nodes that handles the field and attribute
inheritance.  fields and attributes from the parent class are
automatically forwarded to the child.

## Class: EvalContext

**Description:** Holds evaluation time information.  Custom attributes can be attached
to it in extensions.

### Function: get_eval_context(node, ctx)

## Class: Node

**Description:** Baseclass for all Jinja nodes.  There are a number of nodes available
of different types.  There are four major types:

-   :class:`Stmt`: statements
-   :class:`Expr`: expressions
-   :class:`Helper`: helper nodes
-   :class:`Template`: the outermost wrapper node

All nodes have fields and attributes.  Fields may be other nodes, lists,
or arbitrary values.  Fields are passed to the constructor as regular
positional arguments, attributes as keyword arguments.  Each node has
two attributes: `lineno` (the line number of the node) and `environment`.
The `environment` attribute is set at the end of the parsing process for
all nodes automatically.

## Class: Stmt

**Description:** Base node for all statements.

## Class: Helper

**Description:** Nodes that exist in a specific context only.

## Class: Template

**Description:** Node that represents a template.  This must be the outermost node that
is passed to the compiler.

## Class: Output

**Description:** A node that holds multiple expressions which are then printed out.
This is used both for the `print` statement and the regular template data.

## Class: Extends

**Description:** Represents an extends statement.

## Class: For

**Description:** The for loop.  `target` is the target for the iteration (usually a
:class:`Name` or :class:`Tuple`), `iter` the iterable.  `body` is a list
of nodes that are used as loop-body, and `else_` a list of nodes for the
`else` block.  If no else node exists it has to be an empty list.

For filtered nodes an expression can be stored as `test`, otherwise `None`.

## Class: If

**Description:** If `test` is true, `body` is rendered, else `else_`.

## Class: Macro

**Description:** A macro definition.  `name` is the name of the macro, `args` a list of
arguments and `defaults` a list of defaults if there are any.  `body` is
a list of nodes for the macro body.

## Class: CallBlock

**Description:** Like a macro without a name but a call instead.  `call` is called with
the unnamed macro as `caller` argument this node holds.

## Class: FilterBlock

**Description:** Node for filter sections.

## Class: With

**Description:** Specific node for with statements.  In older versions of Jinja the
with statement was implemented on the base of the `Scope` node instead.

.. versionadded:: 2.9.3

## Class: Block

**Description:** A node that represents a block.

.. versionchanged:: 3.0.0
    the `required` field was added.

## Class: Include

**Description:** A node that represents the include tag.

## Class: Import

**Description:** A node that represents the import tag.

## Class: FromImport

**Description:** A node that represents the from import tag.  It's important to not
pass unsafe names to the name attribute.  The compiler translates the
attribute lookups directly into getattr calls and does *not* use the
subscript callback of the interface.  As exported variables may not
start with double underscores (which the parser asserts) this is not a
problem for regular Jinja code, but if this node is used in an extension
extra care must be taken.

The list of names may contain tuples if aliases are wanted.

## Class: ExprStmt

**Description:** A statement that evaluates an expression and discards the result.

## Class: Assign

**Description:** Assigns an expression to a target.

## Class: AssignBlock

**Description:** Assigns a block to a target.

## Class: Expr

**Description:** Baseclass for all expressions.

## Class: BinExpr

**Description:** Baseclass for all binary expressions.

## Class: UnaryExpr

**Description:** Baseclass for all unary expressions.

## Class: Name

**Description:** Looks up a name or stores a value in a name.
The `ctx` of the node can be one of the following values:

-   `store`: store a value in the name
-   `load`: load that name
-   `param`: like `store` but if the name was defined as function parameter.

## Class: NSRef

**Description:** Reference to a namespace value assignment

## Class: Literal

**Description:** Baseclass for literals.

## Class: Const

**Description:** All constant values.  The parser will return this node for simple
constants such as ``42`` or ``"foo"`` but it can be used to store more
complex values such as lists too.  Only constants with a safe
representation (objects where ``eval(repr(x)) == x`` is true).

## Class: TemplateData

**Description:** A constant template string.

## Class: Tuple

**Description:** For loop unpacking and some other things like multiple arguments
for subscripts.  Like for :class:`Name` `ctx` specifies if the tuple
is used for loading the names or storing.

## Class: List

**Description:** Any list literal such as ``[1, 2, 3]``

## Class: Dict

**Description:** Any dict literal such as ``{1: 2, 3: 4}``.  The items must be a list of
:class:`Pair` nodes.

## Class: Pair

**Description:** A key, value pair for dicts.

## Class: Keyword

**Description:** A key, value pair for keyword arguments where key is a string.

## Class: CondExpr

**Description:** A conditional expression (inline if expression).  (``{{
foo if bar else baz }}``)

### Function: args_as_const(node, eval_ctx)

## Class: _FilterTestCommon

## Class: Filter

**Description:** Apply a filter to an expression. ``name`` is the name of the
filter, the other fields are the same as :class:`Call`.

If ``node`` is ``None``, the filter is being used in a filter block
and is applied to the content of the block.

## Class: Test

**Description:** Apply a test to an expression. ``name`` is the name of the test,
the other field are the same as :class:`Call`.

.. versionchanged:: 3.0
    ``as_const`` shares the same logic for filters and tests. Tests
    check for volatile, async, and ``@pass_context`` etc.
    decorators.

## Class: Call

**Description:** Calls an expression.  `args` is a list of arguments, `kwargs` a list
of keyword arguments (list of :class:`Keyword` nodes), and `dyn_args`
and `dyn_kwargs` has to be either `None` or a node that is used as
node for dynamic positional (``*args``) or keyword (``**kwargs``)
arguments.

## Class: Getitem

**Description:** Get an attribute or item from an expression and prefer the item.

## Class: Getattr

**Description:** Get an attribute or item from an expression that is a ascii-only
bytestring and prefer the attribute.

## Class: Slice

**Description:** Represents a slice object.  This must only be used as argument for
:class:`Subscript`.

## Class: Concat

**Description:** Concatenates the list of expressions provided after converting
them to strings.

## Class: Compare

**Description:** Compares an expression with some other expressions.  `ops` must be a
list of :class:`Operand`\s.

## Class: Operand

**Description:** Holds an operator and an expression.

## Class: Mul

**Description:** Multiplies the left with the right node.

## Class: Div

**Description:** Divides the left by the right node.

## Class: FloorDiv

**Description:** Divides the left by the right node and converts the
result into an integer by truncating.

## Class: Add

**Description:** Add the left to the right node.

## Class: Sub

**Description:** Subtract the right from the left node.

## Class: Mod

**Description:** Left modulo right.

## Class: Pow

**Description:** Left to the power of right.

## Class: And

**Description:** Short circuited AND.

## Class: Or

**Description:** Short circuited OR.

## Class: Not

**Description:** Negate the expression.

## Class: Neg

**Description:** Make the expression negative.

## Class: Pos

**Description:** Make the expression positive (noop for most expressions)

## Class: EnvironmentAttribute

**Description:** Loads an attribute from the environment object.  This is useful for
extensions that want to call a callback stored on the environment.

## Class: ExtensionAttribute

**Description:** Returns the attribute of an extension bound to the environment.
The identifier is the identifier of the :class:`Extension`.

This node is usually constructed by calling the
:meth:`~jinja2.ext.Extension.attr` method on an extension.

## Class: ImportedName

**Description:** If created with an import name the import name is returned on node
access.  For example ``ImportedName('cgi.escape')`` returns the `escape`
function from the cgi module on evaluation.  Imports are optimized by the
compiler so there is no need to assign them to local variables.

## Class: InternalName

**Description:** An internal name in the compiler.  You cannot create these nodes
yourself but the parser provides a
:meth:`~jinja2.parser.Parser.free_identifier` method that creates
a new identifier for you.  This identifier is not available from the
template and is not treated specially by the compiler.

## Class: MarkSafe

**Description:** Mark the wrapped expression as safe (wrap it as `Markup`).

## Class: MarkSafeIfAutoescape

**Description:** Mark the wrapped expression as safe (wrap it as `Markup`) but
only if autoescaping is active.

.. versionadded:: 2.5

## Class: ContextReference

**Description:** Returns the current template context.  It can be used like a
:class:`Name` node, with a ``'load'`` ctx and will return the
current :class:`~jinja2.runtime.Context` object.

Here an example that assigns the current template name to a
variable named `foo`::

    Assign(Name('foo', ctx='store'),
           Getattr(ContextReference(), 'name'))

This is basically equivalent to using the
:func:`~jinja2.pass_context` decorator when using the high-level
API, which causes a reference to the context to be passed as the
first argument to a function.

## Class: DerivedContextReference

**Description:** Return the current template context including locals. Behaves
exactly like :class:`ContextReference`, but includes local
variables, such as from a ``for`` loop.

.. versionadded:: 2.11

## Class: Continue

**Description:** Continue a loop.

## Class: Break

**Description:** Break a loop.

## Class: Scope

**Description:** An artificial scope.

## Class: OverlayScope

**Description:** An overlay scope for extensions.  This is a largely unoptimized scope
that however can be used to introduce completely arbitrary variables into
a sub scope from a dictionary or dictionary like object.  The `context`
field has to evaluate to a dictionary object.

Example usage::

    OverlayScope(context=self.call_method('get_context'),
                 body=[...])

.. versionadded:: 2.10

## Class: EvalContextModifier

**Description:** Modifies the eval context.  For each option that should be modified,
a :class:`Keyword` has to be added to the :attr:`options` list.

Example to change the `autoescape` setting::

    EvalContextModifier(options=[Keyword('autoescape', Const(True))])

## Class: ScopedEvalContextModifier

**Description:** Modifies the eval context and reverts it later.  Works exactly like
:class:`EvalContextModifier` but will only modify the
:class:`~jinja2.nodes.EvalContext` for nodes in the :attr:`body`.

### Function: _failing_new()

### Function: __new__(mcs, name, bases, d)

### Function: __init__(self, environment, template_name)

### Function: save(self)

### Function: revert(self, old)

### Function: __init__(self)

### Function: iter_fields(self, exclude, only)

**Description:** This method iterates over all fields that are defined and yields
``(key, value)`` tuples.  Per default all fields are returned, but
it's possible to limit that to some fields by providing the `only`
parameter or to exclude some using the `exclude` parameter.  Both
should be sets or tuples of field names.

### Function: iter_child_nodes(self, exclude, only)

**Description:** Iterates over all direct child nodes of the node.  This iterates
over all fields and yields the values of they are nodes.  If the value
of a field is a list all the nodes in that list are returned.

### Function: find(self, node_type)

**Description:** Find the first node of a given type.  If no such node exists the
return value is `None`.

### Function: find_all(self, node_type)

**Description:** Find all the nodes of a given type.  If the type is a tuple,
the check is performed for any of the tuple items.

### Function: set_ctx(self, ctx)

**Description:** Reset the context of a node and all child nodes.  Per default the
parser will all generate nodes that have a 'load' context as it's the
most common one.  This method is used in the parser to set assignment
targets and other nodes to a store context.

### Function: set_lineno(self, lineno, override)

**Description:** Set the line numbers of the node and children.

### Function: set_environment(self, environment)

**Description:** Set the environment for all nodes.

### Function: __eq__(self, other)

### Function: __repr__(self)

### Function: dump(self)

### Function: as_const(self, eval_ctx)

**Description:** Return the value of the expression as constant or raise
:exc:`Impossible` if this was not possible.

An :class:`EvalContext` can be provided, if none is given
a default context is created which requires the nodes to have
an attached environment.

.. versionchanged:: 2.4
   the `eval_ctx` parameter was added.

### Function: can_assign(self)

**Description:** Check if it's possible to assign something to this node.

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: can_assign(self)

### Function: can_assign(self)

### Function: as_const(self, eval_ctx)

### Function: from_untrusted(cls, value, lineno, environment)

**Description:** Return a const object if the value is representable as
constant value in the generated code, otherwise it will raise
an `Impossible` exception.

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: can_assign(self)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: __init__(self)

### Function: as_const(self, eval_ctx)

### Function: as_const(self, eval_ctx)

### Function: _dump(node)

### Function: const(obj)
