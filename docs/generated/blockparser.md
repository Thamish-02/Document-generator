## AI Summary

A file named blockparser.py.


## Class: State

**Description:** Track the current and nested state of the parser.

This utility class is used to track the state of the `BlockParser` and
support multiple levels if nesting. It's just a simple API wrapped around
a list. Each time a state is set, that state is appended to the end of the
list. Each time a state is reset, that state is removed from the end of
the list.

Therefore, each time a state is set for a nested block, that state must be
reset when we back out of that level of nesting or the state could be
corrupted.

While all the methods of a list object are available, only the three
defined below need be used.

## Class: BlockParser

**Description:** Parse Markdown blocks into an `ElementTree` object.

A wrapper class that stitches the various `BlockProcessors` together,
looping through them and creating an `ElementTree` object.

### Function: set(self, state)

**Description:** Set a new state. 

### Function: reset(self)

**Description:** Step back one step in nested state. 

### Function: isstate(self, state)

**Description:** Test that top (current) level is of given state. 

### Function: __init__(self, md)

**Description:** Initialize the block parser.

Arguments:
    md: A Markdown instance.

Attributes:
    BlockParser.md (Markdown): A Markdown instance.
    BlockParser.state (State): Tracks the nesting level of current location in document being parsed.
    BlockParser.blockprocessors (util.Registry): A collection of
        [`blockprocessors`][markdown.blockprocessors].

### Function: parseDocument(self, lines)

**Description:** Parse a Markdown document into an `ElementTree`.

Given a list of lines, an `ElementTree` object (not just a parent
`Element`) is created and the root element is passed to the parser
as the parent. The `ElementTree` object is returned.

This should only be called on an entire document, not pieces.

Arguments:
    lines: A list of lines (strings).

Returns:
    An element tree.

### Function: parseChunk(self, parent, text)

**Description:** Parse a chunk of Markdown text and attach to given `etree` node.

While the `text` argument is generally assumed to contain multiple
blocks which will be split on blank lines, it could contain only one
block. Generally, this method would be called by extensions when
block parsing is required.

The `parent` `etree` Element passed in is altered in place.
Nothing is returned.

Arguments:
    parent: The parent element.
    text: The text to parse.

### Function: parseBlocks(self, parent, blocks)

**Description:** Process blocks of Markdown text and attach to given `etree` node.

Given a list of `blocks`, each `blockprocessor` is stepped through
until there are no blocks left. While an extension could potentially
call this method directly, it's generally expected to be used
internally.

This is a public method as an extension may need to add/alter
additional `BlockProcessors` which call this method to recursively
parse a nested block.

Arguments:
    parent: The parent element.
    blocks: The blocks of text to parse.
