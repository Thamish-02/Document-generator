## AI Summary

A file named blockprocessors.py.


### Function: build_block_parser(md)

**Description:** Build the default block parser used by Markdown. 

## Class: BlockProcessor

**Description:** Base class for block processors.

Each subclass will provide the methods below to work with the source and
tree. Each processor will need to define it's own `test` and `run`
methods. The `test` method should return True or False, to indicate
whether the current block should be processed by this processor. If the
test passes, the parser will call the processors `run` method.

Attributes:
    BlockProcessor.parser (BlockParser): The `BlockParser` instance this is attached to.
    BlockProcessor.tab_length (int): The tab length set on the `Markdown` instance.

## Class: ListIndentProcessor

**Description:** Process children of list items.

Example

    * a list item
        process this part

        or this part

## Class: CodeBlockProcessor

**Description:** Process code blocks. 

## Class: BlockQuoteProcessor

**Description:** Process blockquotes. 

## Class: OListProcessor

**Description:** Process ordered list blocks. 

## Class: UListProcessor

**Description:** Process unordered list blocks. 

## Class: HashHeaderProcessor

**Description:** Process Hash Headers. 

## Class: SetextHeaderProcessor

**Description:** Process Setext-style Headers. 

## Class: HRProcessor

**Description:** Process Horizontal Rules. 

## Class: EmptyBlockProcessor

**Description:** Process blocks that are empty or start with an empty line. 

## Class: ReferenceProcessor

**Description:** Process link references. 

## Class: ParagraphProcessor

**Description:** Process Paragraph blocks. 

### Function: __init__(self, parser)

### Function: lastChild(self, parent)

**Description:** Return the last child of an `etree` element. 

### Function: detab(self, text, length)

**Description:** Remove a tab from the front of each line of the given text. 

### Function: looseDetab(self, text, level)

**Description:** Remove a tab from front of lines but allowing dedented lines. 

### Function: test(self, parent, block)

**Description:** Test for block type. Must be overridden by subclasses.

As the parser loops through processors, it will call the `test`
method on each to determine if the given block of text is of that
type. This method must return a boolean `True` or `False`. The
actual method of testing is left to the needs of that particular
block type. It could be as simple as `block.startswith(some_string)`
or a complex regular expression. As the block type may be different
depending on the parent of the block (i.e. inside a list), the parent
`etree` element is also provided and may be used as part of the test.

Keyword arguments:
    parent: An `etree` element which will be the parent of the block.
    block: A block of text from the source which has been split at blank lines.

### Function: run(self, parent, blocks)

**Description:** Run processor. Must be overridden by subclasses.

When the parser determines the appropriate type of a block, the parser
will call the corresponding processor's `run` method. This method
should parse the individual lines of the block and append them to
the `etree`.

Note that both the `parent` and `etree` keywords are pointers
to instances of the objects which should be edited in place. Each
processor must make changes to the existing objects as there is no
mechanism to return new/different objects to replace them.

This means that this method should be adding `SubElements` or adding text
to the parent, and should remove (`pop`) or add (`insert`) items to
the list of blocks.

If `False` is returned, this will have the same effect as returning `False`
from the `test` method.

Keyword arguments:
    parent: An `etree` element which is the parent of the current block.
    blocks: A list of all remaining blocks of the document.

### Function: __init__(self)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: create_item(self, parent, block)

**Description:** Create a new `li` and parse the block with it as the parent. 

### Function: get_level(self, parent, block)

**Description:** Get level of indentation based on list level. 

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: clean(self, line)

**Description:** Remove `>` from beginning of a line. 

### Function: __init__(self, parser)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: get_items(self, block)

**Description:** Break a block into list items. 

### Function: __init__(self, parser)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)

### Function: test(self, parent, block)

### Function: run(self, parent, blocks)
