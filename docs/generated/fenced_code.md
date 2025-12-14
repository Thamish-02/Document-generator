## AI Summary

A file named fenced_code.py.


## Class: FencedCodeExtension

## Class: FencedBlockPreprocessor

**Description:** Find and extract fenced code blocks. 

### Function: makeExtension()

### Function: __init__(self)

### Function: extendMarkdown(self, md)

**Description:** Add `FencedBlockPreprocessor` to the Markdown instance. 

### Function: __init__(self, md, config)

### Function: run(self, lines)

**Description:** Match and store Fenced Code Blocks in the `HtmlStash`. 

### Function: handle_attrs(self, attrs)

**Description:** Return tuple: `(id, [list, of, classes], {configs})` 

### Function: _escape(self, txt)

**Description:** basic html escaping 
