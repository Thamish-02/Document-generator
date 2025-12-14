## AI Summary

A file named _htmlparser.py.


## Class: BeautifulSoupHTMLParser

## Class: HTMLParserTreeBuilder

**Description:** A Beautiful soup `bs4.builder.TreeBuilder` that uses the
:py:class:`html.parser.HTMLParser` parser, found in the Python
standard library.

### Function: __init__(self, soup)

### Function: error(self, message)

### Function: handle_startendtag(self, tag, attrs)

**Description:** Handle an incoming empty-element tag.

html.parser only calls this method when the markup looks like
<tag/>.

### Function: handle_starttag(self, tag, attrs, handle_empty_element)

**Description:** Handle an opening tag, e.g. '<tag>'

:param handle_empty_element: True if this tag is known to be
    an empty-element tag (i.e. there is not expected to be any
    closing tag).

### Function: handle_endtag(self, tag, check_already_closed)

**Description:** Handle a closing tag, e.g. '</tag>'

:param tag: A tag name.
:param check_already_closed: True if this tag is expected to
   be the closing portion of an empty-element tag,
   e.g. '<tag></tag>'.

### Function: handle_data(self, data)

**Description:** Handle some textual data that shows up between tags.

### Function: handle_charref(self, name)

**Description:** Handle a numeric character reference by converting it to the
corresponding Unicode character and treating it as textual
data.

:param name: Character number, possibly in hexadecimal.

### Function: handle_entityref(self, name)

**Description:** Handle a named entity reference by converting it to the
corresponding Unicode character(s) and treating it as textual
data.

:param name: Name of the entity reference.

### Function: handle_comment(self, data)

**Description:** Handle an HTML comment.

:param data: The text of the comment.

### Function: handle_decl(self, decl)

**Description:** Handle a DOCTYPE declaration.

:param data: The text of the declaration.

### Function: unknown_decl(self, data)

**Description:** Handle a declaration of unknown type -- probably a CDATA block.

:param data: The text of the declaration.

### Function: handle_pi(self, data)

**Description:** Handle a processing instruction.

:param data: The text of the instruction.

### Function: __init__(self, parser_args, parser_kwargs)

**Description:** Constructor.

:param parser_args: Positional arguments to pass into
    the BeautifulSoupHTMLParser constructor, once it's
    invoked.
:param parser_kwargs: Keyword arguments to pass into
    the BeautifulSoupHTMLParser constructor, once it's
    invoked.
:param kwargs: Keyword arguments for the superclass constructor.

### Function: prepare_markup(self, markup, user_specified_encoding, document_declared_encoding, exclude_encodings)

**Description:** Run any preliminary steps necessary to make incoming markup
acceptable to the parser.

:param markup: Some markup -- probably a bytestring.
:param user_specified_encoding: The user asked to try this encoding.
:param document_declared_encoding: The markup itself claims to be
    in this encoding.
:param exclude_encodings: The user asked _not_ to try any of
    these encodings.

:yield: A series of 4-tuples: (markup, encoding, declared encoding,
     has undergone character replacement)

    Each 4-tuple represents a strategy for parsing the document.
    This TreeBuilder uses Unicode, Dammit to convert the markup
    into Unicode, so the ``markup`` element of the tuple will
    always be a string.

### Function: feed(self, markup)
