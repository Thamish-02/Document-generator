## AI Summary

A file named treeprocessors.py.


### Function: build_treeprocessors(md)

**Description:** Build the default  `treeprocessors` for Markdown. 

### Function: isString(s)

**Description:** Return `True` if object is a string but not an  [`AtomicString`][markdown.util.AtomicString]. 

## Class: Treeprocessor

**Description:** `Treeprocessor`s are run on the `ElementTree` object before serialization.

Each `Treeprocessor` implements a `run` method that takes a pointer to an
`Element` and modifies it as necessary.

`Treeprocessors` must extend `markdown.Treeprocessor`.

## Class: InlineProcessor

**Description:** A `Treeprocessor` that traverses a tree, applying inline patterns.

## Class: PrettifyTreeprocessor

**Description:** Add line breaks to the html document. 

## Class: UnescapeTreeprocessor

**Description:** Restore escaped chars 

### Function: run(self, root)

**Description:** Subclasses of `Treeprocessor` should implement a `run` method, which
takes a root `Element`. This method can return another `Element`
object, and the existing root `Element` will be replaced, or it can
modify the current tree and return `None`.

### Function: __init__(self, md)

### Function: __makePlaceholder(self, type)

**Description:** Generate a placeholder 

### Function: __findPlaceholder(self, data, index)

**Description:** Extract id from data string, start from index.

Arguments:
    data: String.
    index: Index, from which we start search.

Returns:
    Placeholder id and string index, after the found placeholder.

### Function: __stashNode(self, node, type)

**Description:** Add node to stash. 

### Function: __handleInline(self, data, patternIndex)

**Description:** Process string with inline patterns and replace it with placeholders.

Arguments:
    data: A line of Markdown text.
    patternIndex: The index of the `inlinePattern` to start with.

Returns:
    String with placeholders.

### Function: __processElementText(self, node, subnode, isText)

**Description:** Process placeholders in `Element.text` or `Element.tail`
of Elements popped from `self.stashed_nodes`.

Arguments:
    node: Parent node.
    subnode: Processing node.
    isText: Boolean variable, True - it's text, False - it's a tail.

### Function: __processPlaceholders(self, data, parent, isText)

**Description:** Process string with placeholders and generate `ElementTree` tree.

Arguments:
    data: String with placeholders instead of `ElementTree` elements.
    parent: Element, which contains processing inline data.
    isText: Boolean variable, True - it's text, False - it's a tail.

Returns:
    List with `ElementTree` elements with applied inline patterns.

### Function: __applyPattern(self, pattern, data, patternIndex, startIndex)

**Description:** Check if the line fits the pattern, create the necessary
elements, add it to `stashed_nodes`.

Arguments:
    data: The text to be processed.
    pattern: The pattern to be checked.
    patternIndex: Index of current pattern.
    startIndex: String index, from which we start searching.

Returns:
    String with placeholders instead of `ElementTree` elements.

### Function: __build_ancestors(self, parent, parents)

**Description:** Build the ancestor list.

### Function: run(self, tree, ancestors)

**Description:** Apply inline patterns to a parsed Markdown tree.

Iterate over `Element`, find elements with inline tag, apply inline
patterns and append newly created Elements to tree.  To avoid further
processing of string with inline patterns, instead of normal string,
use subclass [`AtomicString`][markdown.util.AtomicString]:

    node.text = markdown.util.AtomicString("This will not be processed.")

Arguments:
    tree: `Element` object, representing Markdown tree.
    ancestors: List of parent tag names that precede the tree node (if needed).

Returns:
    An element tree object with applied inline patterns.

### Function: _prettifyETree(self, elem)

**Description:** Recursively add line breaks to `ElementTree` children. 

### Function: run(self, root)

**Description:** Add line breaks to `Element` object and its children. 

### Function: _unescape(self, m)

### Function: unescape(self, text)

### Function: run(self, root)

**Description:** Loop over all elements and unescape all text. 

### Function: linkText(text)
