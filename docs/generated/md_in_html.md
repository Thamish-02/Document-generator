## AI Summary

A file named md_in_html.py.


## Class: HTMLExtractorExtra

**Description:** Override `HTMLExtractor` and create `etree` `Elements` for any elements which should have content parsed as
Markdown.

## Class: HtmlBlockPreprocessor

**Description:** Remove html blocks from the text and store them for later retrieval.

## Class: MarkdownInHtmlProcessor

**Description:** Process Markdown Inside HTML Blocks which have been stored in the `HtmlStash`.

## Class: MarkdownInHTMLPostprocessor

## Class: MarkdownInHtmlExtension

**Description:** Add Markdown parsing in HTML to Markdown class.

### Function: makeExtension()

### Function: __init__(self, md)

### Function: reset(self)

**Description:** Reset this instance.  Loses all unprocessed data.

### Function: close(self)

**Description:** Handle any buffered data.

### Function: get_element(self)

**Description:** Return element from `treebuilder` and reset `treebuilder` for later use. 

### Function: get_state(self, tag, attrs)

**Description:** Return state from tag and `markdown` attribute. One of 'block', 'span', or 'off'. 

### Function: handle_starttag(self, tag, attrs)

### Function: handle_endtag(self, tag)

### Function: handle_startendtag(self, tag, attrs)

### Function: handle_data(self, data)

### Function: handle_empty_tag(self, data, is_block)

### Function: parse_pi(self, i)

### Function: parse_html_declaration(self, i)

### Function: run(self, lines)

### Function: test(self, parent, block)

### Function: parse_element_content(self, element)

**Description:** Recursively parse the text content of an `etree` Element as Markdown.

Any block level elements generated from the Markdown will be inserted as children of the element in place
of the text content. All `markdown` attributes are removed. For any elements in which Markdown parsing has
been disabled, the text content of it and its children are wrapped in an `AtomicString`.

### Function: run(self, parent, blocks)

### Function: stash_to_string(self, text)

**Description:** Override default to handle any `etree` elements still in the stash. 

### Function: extendMarkdown(self, md)

**Description:** Register extension instances. 
