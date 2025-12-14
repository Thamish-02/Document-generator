## AI Summary

A file named visitors.py.


## Class: _DiscardType

**Description:** When the Discard value is returned from a transformer callback,
that node is discarded and won't appear in the parent.

Note:
    This feature is disabled when the transformer is provided to Lark
    using the ``transformer`` keyword (aka Tree-less LALR mode).

Example:
    ::

        class T(Transformer):
            def ignore_tree(self, children):
                return Discard

            def IGNORE_TOKEN(self, token):
                return Discard

## Class: _Decoratable

**Description:** Provides support for decorating methods with @v_args

## Class: Transformer

**Description:** Transformers work bottom-up (or depth-first), starting with visiting the leaves and working
their way up until ending at the root of the tree.

For each node visited, the transformer will call the appropriate method (callbacks), according to the
node's ``data``, and use the returned value to replace the node, thereby creating a new tree structure.

Transformers can be used to implement map & reduce patterns. Because nodes are reduced from leaf to root,
at any point the callbacks may assume the children have already been transformed (if applicable).

If the transformer cannot find a method with the right name, it will instead call ``__default__``, which by
default creates a copy of the node.

To discard a node, return Discard (``lark.visitors.Discard``).

``Transformer`` can do anything ``Visitor`` can do, but because it reconstructs the tree,
it is slightly less efficient.

A transformer without methods essentially performs a non-memoized partial deepcopy.

All these classes implement the transformer interface:

- ``Transformer`` - Recursively transforms the tree. This is the one you probably want.
- ``Transformer_InPlace`` - Non-recursive. Changes the tree in-place instead of returning new instances
- ``Transformer_InPlaceRecursive`` - Recursive. Changes the tree in-place instead of returning new instances

Parameters:
    visit_tokens (bool, optional): Should the transformer visit tokens in addition to rules.
                                   Setting this to ``False`` is slightly faster. Defaults to ``True``.
                                   (For processing ignored tokens, use the ``lexer_callbacks`` options)

### Function: merge_transformers(base_transformer)

**Description:** Merge a collection of transformers into the base_transformer, each into its own 'namespace'.

When called, it will collect the methods from each transformer, and assign them to base_transformer,
with their name prefixed with the given keyword, as ``prefix__methodname``.

This function is especially useful for processing grammars that import other grammars,
thereby creating some of their rules in a 'namespace'. (i.e with a consistent name prefix).
In this case, the key for the transformer should match the name of the imported grammar.

Parameters:
    base_transformer (Transformer, optional): The transformer that all other transformers will be added to.
    **transformers_to_merge: Keyword arguments, in the form of ``name_prefix = transformer``.

Raises:
    AttributeError: In case of a name collision in the merged methods

Example:
    ::

        class TBase(Transformer):
            def start(self, children):
                return children[0] + 'bar'

        class TImportedGrammar(Transformer):
            def foo(self, children):
                return "foo"

        composed_transformer = merge_transformers(TBase(), imported=TImportedGrammar())

        t = Tree('start', [ Tree('imported__foo', []) ])

        assert composed_transformer.transform(t) == 'foobar'

## Class: InlineTransformer

## Class: TransformerChain

## Class: Transformer_InPlace

**Description:** Same as Transformer, but non-recursive, and changes the tree in-place instead of returning new instances

Useful for huge trees. Conservative in memory.

## Class: Transformer_NonRecursive

**Description:** Same as Transformer but non-recursive.

Like Transformer, it doesn't change the original tree.

Useful for huge trees.

## Class: Transformer_InPlaceRecursive

**Description:** Same as Transformer, recursive, but changes the tree in-place instead of returning new instances

## Class: VisitorBase

## Class: Visitor

**Description:** Tree visitor, non-recursive (can handle huge trees).

Visiting a node calls its methods (provided by the user via inheritance) according to ``tree.data``

## Class: Visitor_Recursive

**Description:** Bottom-up visitor, recursive.

Visiting a node calls its methods (provided by the user via inheritance) according to ``tree.data``

Slightly faster than the non-recursive version.

## Class: Interpreter

**Description:** Interpreter walks the tree starting at the root.

Visits the tree, starting with the root and finally the leaves (top-down)

For each tree node, it calls its methods (provided by user via inheritance) according to ``tree.data``.

Unlike ``Transformer`` and ``Visitor``, the Interpreter doesn't automatically visit its sub-branches.
The user has to explicitly call ``visit``, ``visit_children``, or use the ``@visit_children_decor``.
This allows the user to implement branching and loops.

### Function: visit_children_decor(func)

**Description:** See Interpreter

### Function: _apply_v_args(obj, visit_wrapper)

## Class: _VArgsWrapper

**Description:** A wrapper around a Callable. It delegates `__call__` to the Callable.
If the Callable has a `__get__`, that is also delegate and the resulting function is wrapped.
Otherwise, we use the original function mirroring the behaviour without a __get__.
We also have the visit_wrapper attribute to be used by Transformers.

### Function: _vargs_inline(f, _data, children, _meta)

### Function: _vargs_meta_inline(f, _data, children, meta)

### Function: _vargs_meta(f, _data, children, meta)

### Function: _vargs_tree(f, data, children, meta)

### Function: v_args(inline, meta, tree, wrapper)

**Description:** A convenience decorator factory for modifying the behavior of user-supplied callback methods of ``Transformer`` classes.

By default, transformer callback methods accept one argument - a list of the node's children.

``v_args`` can modify this behavior. When used on a ``Transformer`` class definition, it applies to
all the callback methods inside it.

``v_args`` can be applied to a single method, or to an entire class. When applied to both,
the options given to the method take precedence.

Parameters:
    inline (bool, optional): Children are provided as ``*args`` instead of a list argument (not recommended for very long lists).
    meta (bool, optional): Provides two arguments: ``meta`` and ``children`` (instead of just the latter); ``meta`` isn't available for transformers supplied to Lark using the ``transformer`` parameter (aka internal transformers).
    tree (bool, optional): Provides the entire tree as the argument, instead of the children.
    wrapper (function, optional): Provide a function to decorate all methods.

Example:
    ::

        @v_args(inline=True)
        class SolveArith(Transformer):
            def add(self, left, right):
                return left + right

            @v_args(meta=True)
            def mul(self, meta, children):
                logger.info(f'mul at line {meta.line}')
                left, right = children
                return left * right


        class ReverseNotation(Transformer_InPlace):
            @v_args(tree=True)
            def tree_node(self, tree):
                tree.children = tree.children[::-1]

## Class: CollapseAmbiguities

**Description:** Transforms a tree that contains any number of _ambig nodes into a list of trees,
each one containing an unambiguous tree.

The length of the resulting list is the product of the length of all _ambig nodes.

Warning: This may quickly explode for highly ambiguous trees.

### Function: __repr__(self)

### Function: _apply_v_args(cls, visit_wrapper)

### Function: __class_getitem__(cls, _)

### Function: __init__(self, visit_tokens)

### Function: _call_userfunc(self, tree, new_children)

### Function: _call_userfunc_token(self, token)

### Function: _transform_children(self, children)

### Function: _transform_tree(self, tree)

### Function: transform(self, tree)

**Description:** Transform the given tree, and return the final result

### Function: __mul__(self, other)

**Description:** Chain two transformers together, returning a new transformer.
        

### Function: __default__(self, data, children, meta)

**Description:** Default function that is called if there is no attribute matching ``data``

Can be overridden. Defaults to creating a new copy of the tree node (i.e. ``return Tree(data, children, meta)``)

### Function: __default_token__(self, token)

**Description:** Default function that is called if there is no attribute matching ``token.type``

Can be overridden. Defaults to returning the token as-is.

### Function: _call_userfunc(self, tree, new_children)

### Function: __init__(self)

### Function: transform(self, tree)

### Function: __mul__(self, other)

### Function: _transform_tree(self, tree)

### Function: transform(self, tree)

### Function: transform(self, tree)

### Function: _transform_tree(self, tree)

### Function: _call_userfunc(self, tree)

### Function: __default__(self, tree)

**Description:** Default function that is called if there is no attribute matching ``tree.data``

Can be overridden. Defaults to doing nothing.

### Function: __class_getitem__(cls, _)

### Function: visit(self, tree)

**Description:** Visits the tree, starting with the leaves and finally the root (bottom-up)

### Function: visit_topdown(self, tree)

**Description:** Visit the tree, starting at the root, and ending at the leaves (top-down)

### Function: visit(self, tree)

**Description:** Visits the tree, starting with the leaves and finally the root (bottom-up)

### Function: visit_topdown(self, tree)

**Description:** Visit the tree, starting at the root, and ending at the leaves (top-down)

### Function: visit(self, tree)

### Function: _visit_tree(self, tree)

### Function: visit_children(self, tree)

### Function: __getattr__(self, name)

### Function: __default__(self, tree)

### Function: inner(cls, tree)

### Function: __init__(self, func, visit_wrapper)

### Function: __call__(self)

### Function: __get__(self, instance, owner)

### Function: __set_name__(self, owner, name)

### Function: _visitor_args_dec(obj)

### Function: _ambig(self, options)

### Function: __default__(self, data, children_lists, meta)

### Function: __default_token__(self, t)
