## AI Summary

A file named lxml.py.


## Class: RestrictedElement

**Description:** A restricted Element class that filters out instances of some classes

## Class: GlobalParserTLS

**Description:** Thread local context for custom parser instances

### Function: check_docinfo(elementtree, forbid_dtd, forbid_entities)

**Description:** Check docinfo of an element tree for DTD and entity declarations

The check for entity declarations needs lxml 3 or newer. lxml 2.x does
not support dtd.iterentities().

### Function: parse(source, parser, base_url, forbid_dtd, forbid_entities)

### Function: fromstring(text, parser, base_url, forbid_dtd, forbid_entities)

### Function: iterparse()

### Function: _filter(self, iterator)

### Function: __iter__(self)

### Function: iterchildren(self, tag, reversed)

### Function: iter(self, tag)

### Function: iterdescendants(self, tag)

### Function: itersiblings(self, tag, preceding)

### Function: getchildren(self)

### Function: getiterator(self, tag)

### Function: createDefaultParser(self)

### Function: setDefaultParser(self, parser)

### Function: getDefaultParser(self)
