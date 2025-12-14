## AI Summary

A file named citation.py.


### Function: citation2latex(s)

**Description:** Parse citations in Markdown cells.

This looks for HTML tags having a data attribute names ``data-cite``
and replaces it by the call to LaTeX cite command. The transformation
looks like this::

    <cite data-cite="granger">(Granger, 2013)</cite>

Becomes ::

    \cite{granger}

Any HTML tag can be used, which allows the citations to be formatted
in HTML in any manner.

## Class: CitationParser

**Description:** Citation Parser

Replaces html tags with data-cite attribute with respective latex \cite.

Inherites from HTMLParser, overrides:
 - handle_starttag
 - handle_endtag

### Function: __init__(self)

**Description:** Initialize the parser.

### Function: get_offset(self)

**Description:** Get the offset position.

### Function: handle_starttag(self, tag, attrs)

**Description:** Handle a start tag.

### Function: handle_endtag(self, tag)

**Description:** Handle an end tag.

### Function: feed(self, data)

**Description:** Handle a feed.
