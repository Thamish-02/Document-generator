## AI Summary

A file named _html5lib.py.


## Class: HTML5TreeBuilder

**Description:** Use `html5lib <https://github.com/html5lib/html5lib-python>`_ to
build a tree.

Note that `HTML5TreeBuilder` does not support some common HTML
`TreeBuilder` features. Some of these features could theoretically
be implemented, but at the very least it's quite difficult,
because html5lib moves the parse tree around as it's being built.

Specifically:

* This `TreeBuilder` doesn't use different subclasses of
  `NavigableString` (e.g. `Script`) based on the name of the tag
  in which the string was found.
* You can't use a `SoupStrainer` to parse only part of a document.

## Class: TreeBuilderForHtml5lib

## Class: AttrList

**Description:** Represents a Tag's attributes in a way compatible with html5lib.

## Class: BeautifulSoupNode

## Class: Element

## Class: TextNode

### Function: prepare_markup(self, markup, user_specified_encoding, document_declared_encoding, exclude_encodings)

### Function: feed(self, markup)

**Description:** Run some incoming markup through some parsing process,
populating the `BeautifulSoup` object in `HTML5TreeBuilder.soup`.

### Function: create_treebuilder(self, namespaceHTMLElements)

**Description:** Called by html5lib to instantiate the kind of class it
calls a 'TreeBuilder'.

:param namespaceHTMLElements: Whether or not to namespace HTML elements.

:meta private:

### Function: test_fragment_to_document(self, fragment)

**Description:** See `TreeBuilder`.

### Function: __init__(self, namespaceHTMLElements, soup, store_line_numbers)

### Function: documentClass(self)

### Function: insertDoctype(self, token)

### Function: elementClass(self, name, namespace)

### Function: commentClass(self, data)

### Function: fragmentClass(self)

**Description:** This is only used by html5lib HTMLParser.parseFragment(),
which is never used by Beautiful Soup, only by the html5lib
unit tests. Since we don't currently hook into those tests,
the implementation is left blank.

### Function: getFragment(self)

**Description:** This is only used by the html5lib unit tests. Since we
don't currently hook into those tests, the implementation is
left blank.

### Function: appendChild(self, node)

### Function: getDocument(self)

### Function: testSerializer(self, node)

**Description:** This is only used by the html5lib unit tests. Since we
don't currently hook into those tests, the implementation is
left blank.

### Function: __init__(self, element)

### Function: __iter__(self)

### Function: __setitem__(self, name, value)

### Function: items(self)

### Function: keys(self)

### Function: __len__(self)

### Function: __getitem__(self, name)

### Function: __contains__(self, name)

### Function: element(self)

### Function: nodeType(self)

**Description:** Return the html5lib constant corresponding to the type of
the underlying DOM object.

NOTE: This property is only accessed by the html5lib test
suite, not by Beautiful Soup proper.

### Function: cloneNode(self)

### Function: __init__(self, element, soup, namespace)

### Function: appendChild(self, node)

### Function: getAttributes(self)

### Function: setAttributes(self, attributes)

### Function: insertText(self, data, insertBefore)

### Function: insertBefore(self, node, refNode)

### Function: removeChild(self, node)

### Function: reparentChildren(self, newParent)

**Description:** Move all of this tag's children into another tag.

### Function: hasContent(self)

### Function: cloneNode(self)

### Function: getNameTuple(self)

### Function: __init__(self, element, soup)
