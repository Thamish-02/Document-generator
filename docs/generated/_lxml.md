## AI Summary

A file named _lxml.py.


### Function: _invert(d)

**Description:** Invert a dictionary.

## Class: LXMLTreeBuilderForXML

## Class: LXMLTreeBuilder

### Function: initialize_soup(self, soup)

**Description:** Let the BeautifulSoup object know about the standard namespace
mapping.

:param soup: A `BeautifulSoup`.

### Function: _register_namespaces(self, mapping)

**Description:** Let the BeautifulSoup object know about namespaces encountered
while parsing the document.

This might be useful later on when creating CSS selectors.

This will track (almost) all namespaces, even ones that were
only in scope for part of the document. If two namespaces have
the same prefix, only the first one encountered will be
tracked. Un-prefixed namespaces are not tracked.

:param mapping: A dictionary mapping namespace prefixes to URIs.

### Function: default_parser(self, encoding)

**Description:** Find the default parser for the given encoding.

:return: Either a parser object or a class, which
  will be instantiated with default arguments.

### Function: parser_for(self, encoding)

**Description:** Instantiate an appropriate parser for the given encoding.

:param encoding: A string.
:return: A parser object such as an `etree.XMLParser`.

### Function: __init__(self, parser, empty_element_tags)

### Function: _getNsTag(self, tag)

### Function: prepare_markup(self, markup, user_specified_encoding, document_declared_encoding, exclude_encodings)

**Description:** Run any preliminary steps necessary to make incoming markup
acceptable to the parser.

lxml really wants to get a bytestring and convert it to
Unicode itself. So instead of using UnicodeDammit to convert
the bytestring to Unicode using different encodings, this
implementation uses EncodingDetector to iterate over the
encodings, and tell lxml to try to parse the document as each
one in turn.

:param markup: Some markup -- hopefully a bytestring.
:param user_specified_encoding: The user asked to try this encoding.
:param document_declared_encoding: The markup itself claims to be
    in this encoding.
:param exclude_encodings: The user asked _not_ to try any of
    these encodings.

:yield: A series of 4-tuples: (markup, encoding, declared encoding,
    has undergone character replacement)

    Each 4-tuple represents a strategy for converting the
    document to Unicode and parsing it. Each strategy will be tried
    in turn.

### Function: feed(self, markup)

### Function: close(self)

### Function: start(self, tag, attrib, nsmap)

### Function: _prefix_for_namespace(self, namespace)

**Description:** Find the currently active prefix for the given namespace.

### Function: end(self, tag)

### Function: pi(self, target, data)

### Function: data(self, data)

### Function: doctype(self, name, pubid, system)

### Function: comment(self, text)

**Description:** Handle comments as Comment objects.

### Function: test_fragment_to_document(self, fragment)

**Description:** See `TreeBuilder`.

### Function: default_parser(self, encoding)

### Function: feed(self, markup)

### Function: test_fragment_to_document(self, fragment)

**Description:** See `TreeBuilder`.
