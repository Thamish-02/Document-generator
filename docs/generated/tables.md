## AI Summary

A file named tables.py.


## Class: TableProcessor

**Description:** Process Tables. 

## Class: TableExtension

**Description:** Add tables to Markdown. 

### Function: makeExtension()

### Function: __init__(self, parser, config)

### Function: test(self, parent, block)

**Description:** Ensure first two rows (column header and separator row) are valid table rows.

Keep border check and separator row do avoid repeating the work.

### Function: run(self, parent, blocks)

**Description:** Parse a table block and build table. 

### Function: _build_empty_row(self, parent, align)

**Description:** Build an empty row.

### Function: _build_row(self, row, parent, align)

**Description:** Given a row of text, build table cells. 

### Function: _split_row(self, row)

**Description:** split a row of text into list of cells. 

### Function: _split(self, row)

**Description:** split a row of text with some code into a list of cells. 

### Function: __init__(self)

### Function: extendMarkdown(self, md)

**Description:** Add an instance of `TableProcessor` to `BlockParser`. 
