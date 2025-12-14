## AI Summary

A file named ast_utils.py.


## Class: Ast

**Description:** Abstract class

Subclasses will be collected by `create_transformer()`

## Class: AsList

**Description:** Abstract class

Subclasses will be instantiated with the parse results as a single list, instead of as arguments.

## Class: WithMeta

**Description:** Abstract class

Subclasses will be instantiated with the Meta instance of the tree. (see ``v_args`` for more detail)

### Function: camel_to_snake(name)

### Function: create_transformer(ast_module, transformer, decorator_factory)

**Description:** Collects `Ast` subclasses from the given module, and creates a Lark transformer that builds the AST.

For each class, we create a corresponding rule in the transformer, with a matching name.
CamelCase names will be converted into snake_case. Example: "CodeBlock" -> "code_block".

Classes starting with an underscore (`_`) will be skipped.

Parameters:
    ast_module: A Python module containing all the subclasses of ``ast_utils.Ast``
    transformer (Optional[Transformer]): An initial transformer. Its attributes may be overwritten.
    decorator_factory (Callable): An optional callable accepting two booleans, inline, and meta,
        and returning a decorator for the methods of ``transformer``. (default: ``v_args``).
