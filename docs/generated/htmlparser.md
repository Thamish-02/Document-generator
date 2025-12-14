## AI Summary

A file named htmlparser.py.


## Class: HTMLExtractor

**Description:** Extract raw HTML from text.

The raw HTML is stored in the [`htmlStash`][markdown.util.HtmlStash] of the
[`Markdown`][markdown.Markdown] instance passed to `md` and the remaining text
is stored in `cleandoc` as a list of strings.

### Function: __init__(self, md)

### Function: reset(self)

**Description:** Reset this instance.  Loses all unprocessed data.

### Function: close(self)

**Description:** Handle any buffered data.

### Function: line_offset(self)

**Description:** Returns char index in `self.rawdata` for the start of the current line. 

### Function: at_line_start(self)

**Description:** Returns True if current position is at start of line.

Allows for up to three blank spaces at start of line.

### Function: get_endtag_text(self, tag)

**Description:** Returns the text of the end tag.

If it fails to extract the actual text from the raw data, it builds a closing tag with `tag`.

### Function: handle_starttag(self, tag, attrs)

### Function: handle_endtag(self, tag)

### Function: handle_data(self, data)

### Function: handle_empty_tag(self, data, is_block)

**Description:** Handle empty tags (`<data>`). 

### Function: handle_startendtag(self, tag, attrs)

### Function: handle_charref(self, name)

### Function: handle_entityref(self, name)

### Function: handle_comment(self, data)

### Function: updatepos(self, i, j)

### Function: handle_decl(self, data)

### Function: handle_pi(self, data)

### Function: unknown_decl(self, data)

### Function: parse_pi(self, i)

### Function: parse_html_declaration(self, i)

### Function: parse_bogus_comment(self, i, report)

### Function: get_starttag_text(self)

**Description:** Return full source of start tag: `<...>`.

### Function: parse_starttag(self, i)

### Function: parse_comment(self, i, report)
