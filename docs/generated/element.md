## AI Summary

A file named element.py.


### Function: __getattr__(name)

## Class: NamespacedAttribute

**Description:** A namespaced attribute (e.g. the 'xml:lang' in 'xml:lang="en"')
which remembers the namespace prefix ('xml') and the name ('lang')
that were used to create it.

## Class: AttributeValueWithCharsetSubstitution

**Description:** An abstract class standing in for a character encoding specified
inside an HTML ``<meta>`` tag.

Subclasses exist for each place such a character encoding might be
found: either inside the ``charset`` attribute
(`CharsetMetaAttributeValue`) or inside the ``content`` attribute
(`ContentMetaAttributeValue`)

This allows Beautiful Soup to replace that part of the HTML file
with a different encoding when ouputting a tree as a string.

## Class: CharsetMetaAttributeValue

**Description:** A generic stand-in for the value of a ``<meta>`` tag's ``charset``
attribute.

When Beautiful Soup parses the markup ``<meta charset="utf8">``, the
value of the ``charset`` attribute will become one of these objects.

If the document is later encoded to an encoding other than UTF-8, its
``<meta>`` tag will mention the new encoding instead of ``utf8``.

## Class: AttributeValueList

**Description:** Class for the list used to hold the values of attributes which
have multiple values (such as HTML's 'class'). It's just a regular
list, but you can subclass it and pass it in to the TreeBuilder
constructor as attribute_value_list_class, to have your subclass
instantiated instead.

## Class: AttributeDict

**Description:** Superclass for the dictionary used to hold a tag's
attributes. You can use this, but it's just a regular dict with no
special logic.

## Class: XMLAttributeDict

**Description:** A dictionary for holding a Tag's attributes, which processes
incoming values for consistency with the HTML spec.

## Class: HTMLAttributeDict

**Description:** A dictionary for holding a Tag's attributes, which processes
incoming values for consistency with the HTML spec, which says
'Attribute values are a mixture of text and character
references...'

Basically, this means converting common non-string values into
strings, like XMLAttributeDict, though HTML also has some rules
around boolean attributes that XML doesn't have.

## Class: ContentMetaAttributeValue

**Description:** A generic stand-in for the value of a ``<meta>`` tag's ``content``
attribute.

When Beautiful Soup parses the markup:
 ``<meta http-equiv="content-type" content="text/html; charset=utf8">``

The value of the ``content`` attribute will become one of these objects.

If the document is later encoded to an encoding other than UTF-8, its
``<meta>`` tag will mention the new encoding instead of ``utf8``.

## Class: PageElement

**Description:** An abstract class representing a single element in the parse tree.

`NavigableString`, `Tag`, etc. are all subclasses of
`PageElement`. For this reason you'll see a lot of methods that
return `PageElement`, but you'll never see an actual `PageElement`
object. For the most part you can think of `PageElement` as
meaning "a `Tag` or a `NavigableString`."

## Class: NavigableString

**Description:** A Python string that is part of a parse tree.

When Beautiful Soup parses the markup ``<b>penguin</b>``, it will
create a `NavigableString` for the string "penguin".

## Class: PreformattedString

**Description:** A `NavigableString` not subject to the normal formatting rules.

This is an abstract class used for special kinds of strings such
as comments (`Comment`) and CDATA blocks (`CData`).

## Class: CData

**Description:** A `CDATA section <https://dev.w3.org/html5/spec-LC/syntax.html#cdata-sections>`_.

## Class: ProcessingInstruction

**Description:** A SGML processing instruction.

## Class: XMLProcessingInstruction

**Description:** An `XML processing instruction <https://www.w3.org/TR/REC-xml/#sec-pi>`_.

## Class: Comment

**Description:** An `HTML comment <https://dev.w3.org/html5/spec-LC/syntax.html#comments>`_ or `XML comment <https://www.w3.org/TR/REC-xml/#sec-comments>`_.

## Class: Declaration

**Description:** An `XML declaration <https://www.w3.org/TR/REC-xml/#sec-prolog-dtd>`_.

## Class: Doctype

**Description:** A `document type declaration <https://www.w3.org/TR/REC-xml/#dt-doctype>`_.

## Class: Stylesheet

**Description:** A `NavigableString` representing the contents of a `<style> HTML
tag <https://dev.w3.org/html5/spec-LC/Overview.html#the-style-element>`_
(probably CSS).

Used to distinguish embedded stylesheets from textual content.

## Class: Script

**Description:** A `NavigableString` representing the contents of a `<script>
HTML tag
<https://dev.w3.org/html5/spec-LC/Overview.html#the-script-element>`_
(probably Javascript).

Used to distinguish executable code from textual content.

## Class: TemplateString

**Description:** A `NavigableString` representing a string found inside an `HTML
<template> tag <https://html.spec.whatwg.org/multipage/scripting.html#the-template-element>`_
embedded in a larger document.

Used to distinguish such strings from the main body of the document.

## Class: RubyTextString

**Description:** A NavigableString representing the contents of an `<rt> HTML
tag <https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rt-element>`_.

Can be used to distinguish such strings from the strings they're
annotating.

## Class: RubyParenthesisString

**Description:** A NavigableString representing the contents of an `<rp> HTML
tag <https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rp-element>`_.

## Class: Tag

**Description:** An HTML or XML tag that is part of a parse tree, along with its
attributes, contents, and relationships to other parts of the tree.

When Beautiful Soup parses the markup ``<b>penguin</b>``, it will
create a `Tag` object representing the ``<b>`` tag. You can
instantiate `Tag` objects directly, but it's not necessary unless
you're adding entirely new markup to a parsed document. Most of
the constructor arguments are intended for use by the `TreeBuilder`
that's parsing a document.

:param parser: A `BeautifulSoup` object representing the parse tree this
    `Tag` will be part of.
:param builder: The `TreeBuilder` being used to build the tree.
:param name: The name of the tag.
:param namespace: The URI of this tag's XML namespace, if any.
:param prefix: The prefix for this tag's XML namespace, if any.
:param attrs: A dictionary of attribute values.
:param parent: The `Tag` to use as the parent of this `Tag`. May be
   the `BeautifulSoup` object itself.
:param previous: The `PageElement` that was parsed immediately before
    parsing this tag.
:param is_xml: If True, this is an XML tag. Otherwise, this is an
    HTML tag.
:param sourceline: The line number where this tag was found in its
    source document.
:param sourcepos: The character position within ``sourceline`` where this
    tag was found.
:param can_be_empty_element: If True, this tag should be
    represented as <tag/>. If False, this tag should be represented
    as <tag></tag>.
:param cdata_list_attributes: A dictionary of attributes whose values should
    be parsed as lists of strings if they ever show up on this tag.
:param preserve_whitespace_tags: Names of tags whose contents
    should have their whitespace preserved if they are encountered inside
    this tag.
:param interesting_string_types: When iterating over this tag's
    string contents in methods like `Tag.strings` or
    `PageElement.get_text`, these are the types of strings that are
    interesting enough to be considered. By default,
    `NavigableString` (normal strings) and `CData` (CDATA
    sections) are the only interesting string subtypes.
:param namespaces: A dictionary mapping currently active
    namespace prefixes to URIs, as of the point in the parsing process when
    this tag was encountered. This can be used later to
    construct CSS selectors.

## Class: ResultSet

**Description:** A ResultSet is a list of `PageElement` objects, gathered as the result
of matching an :py:class:`ElementFilter` against a parse tree. Basically, a list of
search results.

### Function: __new__(cls, prefix, name, namespace)

### Function: substitute_encoding(self, eventual_encoding)

**Description:** Do whatever's necessary in this implementation-specific
portion an HTML document to substitute in a specific encoding.

### Function: __new__(cls, original_value)

### Function: substitute_encoding(self, eventual_encoding)

**Description:** When an HTML document is being encoded to a given encoding, the
value of a ``<meta>`` tag's ``charset`` becomes the name of
the encoding.

### Function: __setitem__(self, key, value)

**Description:** Set an attribute value, possibly modifying it to comply with
the XML spec.

This just means converting common non-string values to
strings: XML attributes may have "any literal string as a
value."

### Function: __setitem__(self, key, value)

**Description:** Set an attribute value, possibly modifying it to comply
with the HTML spec,

### Function: __new__(cls, original_value)

### Function: substitute_encoding(self, eventual_encoding)

**Description:** When an HTML document is being encoded to a given encoding, the
value of the ``charset=`` in a ``<meta>`` tag's ``content`` becomes
the name of the encoding.

### Function: setup(self, parent, previous_element, next_element, previous_sibling, next_sibling)

**Description:** Sets up the initial relations between this element and
other elements.

:param parent: The parent of this element.

:param previous_element: The element parsed immediately before
    this one.

:param next_element: The element parsed immediately after
    this one.

:param previous_sibling: The most recently encountered element
    on the same level of the parse tree as this one.

:param previous_sibling: The next element to be encountered
    on the same level of the parse tree as this one.

### Function: format_string(self, s, formatter)

**Description:** Format the given string using the given formatter.

:param s: A string.
:param formatter: A Formatter object, or a string naming one of the standard formatters.

### Function: formatter_for_name(self, formatter_name)

**Description:** Look up or create a Formatter for the given identifier,
if necessary.

:param formatter: Can be a `Formatter` object (used as-is), a
    function (used as the entity substitution hook for an
    `bs4.formatter.XMLFormatter` or
    `bs4.formatter.HTMLFormatter`), or a string (used to look
    up an `bs4.formatter.XMLFormatter` or
    `bs4.formatter.HTMLFormatter` in the appropriate registry.

### Function: _is_xml(self)

**Description:** Is this element part of an XML tree or an HTML tree?

This is used in formatter_for_name, when deciding whether an
XMLFormatter or HTMLFormatter is more appropriate. It can be
inefficient, but it should be called very rarely.

### Function: __deepcopy__(self, memo, recursive)

### Function: __copy__(self)

**Description:** A copy of a PageElement can only be a deep copy, because
only one PageElement can occupy a given place in a parse tree.

### Function: _all_strings(self, strip, types)

**Description:** Yield all strings of certain classes, possibly stripping them.

This is implemented differently in `Tag` and `NavigableString`.

### Function: stripped_strings(self)

**Description:** Yield all interesting strings in this PageElement, stripping them
first.

See `Tag` for information on which strings are considered
interesting in a given context.

### Function: get_text(self, separator, strip, types)

**Description:** Get all child strings of this PageElement, concatenated using the
given separator.

:param separator: Strings will be concatenated using this separator.

:param strip: If True, strings will be stripped before being
    concatenated.

:param types: A tuple of NavigableString subclasses. Any
    strings of a subclass not found in this list will be
    ignored. Although there are exceptions, the default
    behavior in most cases is to consider only NavigableString
    and CData objects. That means no comments, processing
    instructions, etc.

:return: A string.

### Function: replace_with(self)

**Description:** Replace this `PageElement` with one or more other elements,
objects, keeping the rest of the tree the same.

:return: This `PageElement`, no longer part of the tree.

### Function: wrap(self, wrap_inside)

**Description:** Wrap this `PageElement` inside a `Tag`.

:return: ``wrap_inside``, occupying the position in the tree that used
   to be occupied by this object, and with this object now inside it.

### Function: extract(self, _self_index)

**Description:** Destructively rips this element out of the tree.

:param _self_index: The location of this element in its parent's
   .contents, if known. Passing this in allows for a performance
   optimization.

:return: this `PageElement`, no longer part of the tree.

### Function: decompose(self)

**Description:** Recursively destroys this `PageElement` and its children.

The element will be removed from the tree and wiped out; so
will everything beneath it.

The behavior of a decomposed `PageElement` is undefined and you
should never use one for anything, but if you need to *check*
whether an element has been decomposed, you can use the
`PageElement.decomposed` property.

### Function: _last_descendant(self, is_initialized, accept_self)

**Description:** Finds the last element beneath this object to be parsed.

Special note to help you figure things out if your type
checking is tripped up by the fact that this method returns
_AtMostOneElement instead of PageElement: the only time
this method returns None is if `accept_self` is False and the
`PageElement` has no children--either it's a NavigableString
or an empty Tag.

:param is_initialized: Has `PageElement.setup` been called on
    this `PageElement` yet?

:param accept_self: Is ``self`` an acceptable answer to the
    question?

### Function: insert_before(self)

**Description:** Makes the given element(s) the immediate predecessor of this one.

All the elements will have the same `PageElement.parent` as
this one, and the given elements will occur immediately before
this one.

:param args: One or more PageElements.

:return The list of PageElements that were inserted.

### Function: insert_after(self)

**Description:** Makes the given element(s) the immediate successor of this one.

The elements will have the same `PageElement.parent` as this
one, and the given elements will occur immediately after this
one.

:param args: One or more PageElements.

:return The list of PageElements that were inserted.

### Function: find_next(self, name, attrs, string)

### Function: find_next(self, name, attrs, string)

### Function: find_next(self, name, attrs, string)

**Description:** Find the first PageElement that matches the given criteria and
appears later in the document than this PageElement.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a NavigableString with specific text.
:kwargs: Additional filters on attribute values.

### Function: find_all_next(self, name, attrs, string, limit, _stacklevel)

### Function: find_all_next(self, name, attrs, string, limit, _stacklevel)

### Function: find_all_next(self, name, attrs, string, limit, _stacklevel)

**Description:** Find all `PageElement` objects that match the given criteria and
appear later in the document than this `PageElement`.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a NavigableString with specific text.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: find_next_sibling(self, name, attrs, string)

### Function: find_next_sibling(self, name, attrs, string)

### Function: find_next_sibling(self, name, attrs, string)

**Description:** Find the closest sibling to this PageElement that matches the
given criteria and appears later in the document.

All find_* methods take a common set of arguments. See the
online documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a `NavigableString` with specific text.
:kwargs: Additional filters on attribute values.

### Function: find_next_siblings(self, name, attrs, string, limit, _stacklevel)

### Function: find_next_siblings(self, name, attrs, string, limit, _stacklevel)

### Function: find_next_siblings(self, name, attrs, string, limit, _stacklevel)

**Description:** Find all siblings of this `PageElement` that match the given criteria
and appear later in the document.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a `NavigableString` with specific text.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: find_previous(self, name, attrs, string)

### Function: find_previous(self, name, attrs, string)

### Function: find_previous(self, name, attrs, string)

**Description:** Look backwards in the document from this `PageElement` and find the
first `PageElement` that matches the given criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a `NavigableString` with specific text.
:kwargs: Additional filters on attribute values.

### Function: find_all_previous(self, name, attrs, string, limit, _stacklevel)

### Function: find_all_previous(self, name, attrs, string, limit, _stacklevel)

### Function: find_all_previous(self, name, attrs, string, limit, _stacklevel)

**Description:** Look backwards in the document from this `PageElement` and find all
`PageElement` that match the given criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a `NavigableString` with specific text.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: find_previous_sibling(self, name, attrs, string)

### Function: find_previous_sibling(self, name, attrs, string)

### Function: find_previous_sibling(self, name, attrs, string)

**Description:** Returns the closest sibling to this `PageElement` that matches the
given criteria and appears earlier in the document.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a `NavigableString` with specific text.
:kwargs: Additional filters on attribute values.

### Function: find_previous_siblings(self, name, attrs, string, limit, _stacklevel)

### Function: find_previous_siblings(self, name, attrs, string, limit, _stacklevel)

### Function: find_previous_siblings(self, name, attrs, string, limit, _stacklevel)

**Description:** Returns all siblings to this PageElement that match the
given criteria and appear earlier in the document.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param string: A filter for a NavigableString with specific text.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: find_parent(self, name, attrs)

**Description:** Find the closest parent of this PageElement that matches the given
criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param self: Whether the PageElement itself should be considered
   as one of its 'parents'.
:kwargs: Additional filters on attribute values.

### Function: find_parents(self, name, attrs, limit, _stacklevel)

**Description:** Find all parents of this `PageElement` that match the given criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: next(self)

**Description:** The `PageElement`, if any, that was parsed just after this one.

### Function: previous(self)

**Description:** The `PageElement`, if any, that was parsed just before this one.

### Function: _find_one(self, method, name, attrs, string)

### Function: _find_all(self, name, attrs, string, limit, generator, _stacklevel)

**Description:** Iterates over a generator looking for things that match.

### Function: next_elements(self)

**Description:** All PageElements that were parsed after this one.

### Function: self_and_next_elements(self)

**Description:** This PageElement, then all PageElements that were parsed after it.

### Function: next_siblings(self)

**Description:** All PageElements that are siblings of this one but were parsed
later.

### Function: self_and_next_siblings(self)

**Description:** This PageElement, then all of its siblings.

### Function: previous_elements(self)

**Description:** All PageElements that were parsed before this one.

:yield: A sequence of PageElements.

### Function: self_and_previous_elements(self)

**Description:** This PageElement, then all elements that were parsed
earlier.

### Function: previous_siblings(self)

**Description:** All PageElements that are siblings of this one but were parsed
earlier.

:yield: A sequence of PageElements.

### Function: self_and_previous_siblings(self)

**Description:** This PageElement, then all of its siblings that were parsed
earlier.

### Function: parents(self)

**Description:** All elements that are parents of this PageElement.

:yield: A sequence of Tags, ending with a BeautifulSoup object.

### Function: self_and_parents(self)

**Description:** This element, then all of its parents.

:yield: A sequence of PageElements, ending with a BeautifulSoup object.

### Function: _self_and(self, other_generator)

**Description:** Modify a generator by yielding this element, then everything
yielded by the other generator.

### Function: decomposed(self)

**Description:** Check whether a PageElement has been decomposed.

### Function: nextGenerator(self)

**Description:** :meta private:

### Function: nextSiblingGenerator(self)

**Description:** :meta private:

### Function: previousGenerator(self)

**Description:** :meta private:

### Function: previousSiblingGenerator(self)

**Description:** :meta private:

### Function: parentGenerator(self)

**Description:** :meta private:

### Function: __new__(cls, value)

**Description:** Create a new NavigableString.

When unpickling a NavigableString, this method is called with
the string in DEFAULT_OUTPUT_ENCODING. That encoding needs to be
passed in to the superclass's __new__ or the superclass won't know
how to handle non-ASCII characters.

### Function: __deepcopy__(self, memo, recursive)

**Description:** A copy of a NavigableString has the same contents and class
as the original, but it is not connected to the parse tree.

:param recursive: This parameter is ignored; it's only defined
   so that NavigableString.__deepcopy__ implements the same
   signature as Tag.__deepcopy__.

### Function: __getnewargs__(self)

### Function: __getitem__(self, key)

**Description:** Raise an exception 

### Function: string(self)

**Description:** Convenience property defined to match `Tag.string`.

:return: This property always returns the `NavigableString` it was
   called on.

:meta private:

### Function: output_ready(self, formatter)

**Description:** Run the string through the provided formatter, making it
ready for output as part of an HTML or XML document.

:param formatter: A `Formatter` object, or a string naming one
    of the standard formatters.

### Function: name(self)

**Description:** Since a NavigableString is not a Tag, it has no .name.

This property is implemented so that code like this doesn't crash
when run on a mixture of Tag and NavigableString objects:
    [x.name for x in tag.children]

:meta private:

### Function: name(self, name)

**Description:** Prevent NavigableString.name from ever being set.

:meta private:

### Function: _all_strings(self, strip, types)

**Description:** Yield all strings of certain classes, possibly stripping them.

This makes it easy for NavigableString to implement methods
like get_text() as conveniences, creating a consistent
text-extraction API across all PageElements.

:param strip: If True, all strings will be stripped before being
    yielded.

:param types: A tuple of NavigableString subclasses. If this
    NavigableString isn't one of those subclasses, the
    sequence will be empty. By default, the subclasses
    considered are NavigableString and CData objects. That
    means no comments, processing instructions, etc.

:yield: A sequence that either contains this string, or is empty.

### Function: strings(self)

**Description:** Yield this string, but only if it is interesting.

This is defined the way it is for compatibility with
`Tag.strings`. See `Tag` for information on which strings are
interesting in a given context.

:yield: A sequence that either contains this string, or is empty.

### Function: output_ready(self, formatter)

**Description:** Make this string ready for output by adding any subclass-specific
    prefix or suffix.

:param formatter: A `Formatter` object, or a string naming one
    of the standard formatters. The string will be passed into the
    `Formatter`, but only to trigger any side effects: the return
    value is ignored.

:return: The string, with any subclass-specific prefix and
   suffix added on.

### Function: for_name_and_ids(cls, name, pub_id, system_id)

**Description:** Generate an appropriate document type declaration for a given
public ID and system ID.

:param name: The name of the document's root element, e.g. 'html'.
:param pub_id: The Formal Public Identifier for this document type,
    e.g. '-//W3C//DTD XHTML 1.1//EN'
:param system_id: The system identifier for this document type,
    e.g. 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'

### Function: _string_for_name_and_ids(cls, name, pub_id, system_id)

**Description:** Generate a string to be used as the basis of a Doctype object.

This is a separate method from for_name_and_ids() because the lxml
TreeBuilder needs to call it.

### Function: __init__(self, parser, builder, name, namespace, prefix, attrs, parent, previous, is_xml, sourceline, sourcepos, can_be_empty_element, cdata_list_attributes, preserve_whitespace_tags, interesting_string_types, namespaces)

### Function: __deepcopy__(self, memo, recursive)

**Description:** A deepcopy of a Tag is a new Tag, unconnected to the parse tree.
Its contents are a copy of the old Tag's contents.

### Function: copy_self(self)

**Description:** Create a new Tag just like this one, but with no
contents and unattached to any parse tree.

This is the first step in the deepcopy process, but you can
call it on its own to create a copy of a Tag without copying its
contents.

### Function: is_empty_element(self)

**Description:** Is this tag an empty-element tag? (aka a self-closing tag)

A tag that has contents is never an empty-element tag.

A tag that has no contents may or may not be an empty-element
tag. It depends on the `TreeBuilder` used to create the
tag. If the builder has a designated list of empty-element
tags, then only a tag whose name shows up in that list is
considered an empty-element tag. This is usually the case
for HTML documents.

If the builder has no designated list of empty-element, then
any tag with no contents is an empty-element tag. This is usually
the case for XML documents.

### Function: isSelfClosing(self)

**Description:** : :meta private:

### Function: string(self)

**Description:** Convenience property to get the single string within this
`Tag`, assuming there is just one.

:return: If this `Tag` has a single child that's a
 `NavigableString`, the return value is that string. If this
 element has one child `Tag`, the return value is that child's
 `Tag.string`, recursively. If this `Tag` has no children,
 or has more than one child, the return value is ``None``.

 If this property is unexpectedly returning ``None`` for you,
 it's probably because your `Tag` has more than one thing
 inside it.

### Function: string(self, string)

**Description:** Replace the `Tag.contents` of this `Tag` with a single string.

### Function: _all_strings(self, strip, types)

**Description:** Yield all strings of certain classes, possibly stripping them.

:param strip: If True, all strings will be stripped before being
    yielded.

:param types: A tuple of NavigableString subclasses. Any strings of
    a subclass not found in this list will be ignored. By
    default, the subclasses considered are the ones found in
    self.interesting_string_types. If that's not specified,
    only NavigableString and CData objects will be
    considered. That means no comments, processing
    instructions, etc.

### Function: insert(self, position)

**Description:** Insert one or more new PageElements as a child of this `Tag`.

This works similarly to :py:meth:`list.insert`, except you can insert
multiple elements at once.

:param position: The numeric position that should be occupied
   in this Tag's `Tag.children` by the first new `PageElement`.

:param new_children: The PageElements to insert.

:return The newly inserted PageElements.

### Function: _insert(self, position, new_child)

### Function: unwrap(self)

**Description:** Replace this `PageElement` with its contents.

:return: This object, no longer part of the tree.

### Function: replaceWithChildren(self)

**Description:** : :meta private:

### Function: append(self, tag)

**Description:** Appends the given `PageElement` to the contents of this `Tag`.

:param tag: A PageElement.

:return The newly appended PageElement.

### Function: extend(self, tags)

**Description:** Appends one or more objects to the contents of this
`Tag`.

:param tags: If a list of `PageElement` objects is provided,
    they will be appended to this tag's contents, one at a time.
    If a single `Tag` is provided, its `Tag.contents` will be
    used to extend this object's `Tag.contents`.

:return The list of PageElements that were appended.

### Function: clear(self, decompose)

**Description:** Destroy all children of this `Tag` by calling
   `PageElement.extract` on them.

:param decompose: If this is True, `PageElement.decompose` (a
    more destructive method) will be called instead of
    `PageElement.extract`.

### Function: smooth(self)

**Description:** Smooth out the children of this `Tag` by consolidating consecutive
strings.

If you perform a lot of operations that modify the tree,
calling this method afterwards can make pretty-printed output
look more natural.

### Function: index(self, element)

**Description:** Find the index of a child of this `Tag` (by identity, not value).

Doing this by identity avoids issues when a `Tag` contains two
children that have string equality.

:param element: Look for this `PageElement` in this object's contents.

### Function: get(self, key, default)

**Description:** Returns the value of the 'key' attribute for the tag, or
the value given for 'default' if it doesn't have that
attribute.

:param key: The attribute to look for.
:param default: Use this value if the attribute is not present
    on this `Tag`.

### Function: get_attribute_list(self, key, default)

**Description:** The same as get(), but always returns a (possibly empty) list.

:param key: The attribute to look for.
:param default: Use this value if the attribute is not present
    on this `Tag`.
:return: A list of strings, usually empty or containing only a single
    value.

### Function: has_attr(self, key)

**Description:** Does this `Tag` have an attribute with the given name?

### Function: __hash__(self)

### Function: __getitem__(self, key)

**Description:** tag[key] returns the value of the 'key' attribute for the Tag,
and throws an exception if it's not there.

### Function: __iter__(self)

**Description:** Iterating over a Tag iterates over its contents.

### Function: __len__(self)

**Description:** The length of a Tag is the length of its list of contents.

### Function: __contains__(self, x)

### Function: __bool__(self)

**Description:** A tag is non-None even if it has no contents.

### Function: __setitem__(self, key, value)

**Description:** Setting tag[key] sets the value of the 'key' attribute for the
tag.

### Function: __delitem__(self, key)

**Description:** Deleting tag[key] deletes all 'key' attributes for the tag.

### Function: __call__(self, name, attrs, recursive, string, limit, _stacklevel)

### Function: __call__(self, name, attrs, recursive, string, limit, _stacklevel)

### Function: __call__(self, name, attrs, recursive, string, limit, _stacklevel)

**Description:** Calling a Tag like a function is the same as calling its
find_all() method. Eg. tag('a') returns a list of all the A tags
found within this tag.

### Function: __getattr__(self, subtag)

**Description:** Calling tag.subtag is the same as calling tag.find(name="subtag")

### Function: __eq__(self, other)

**Description:** Returns true iff this Tag has the same name, the same attributes,
and the same contents (recursively) as `other`.

### Function: __ne__(self, other)

**Description:** Returns true iff this Tag is not identical to `other`,
as defined in __eq__.

### Function: __repr__(self)

**Description:** Renders this `Tag` as a string.

### Function: encode(self, encoding, indent_level, formatter, errors)

**Description:** Render this `Tag` and its contents as a bytestring.

:param encoding: The encoding to use when converting to
   a bytestring. This may also affect the text of the document,
   specifically any encoding declarations within the document.
:param indent_level: Each line of the rendering will be
   indented this many levels. (The ``formatter`` decides what a
   'level' means, in terms of spaces or other characters
   output.) This is used internally in recursive calls while
   pretty-printing.
:param formatter: Either a `Formatter` object, or a string naming one of
    the standard formatters.
:param errors: An error handling strategy such as
    'xmlcharrefreplace'. This value is passed along into
    :py:meth:`str.encode` and its value should be one of the `error
    handling constants defined by Python's codecs module
    <https://docs.python.org/3/library/codecs.html#error-handlers>`_.

### Function: decode(self, indent_level, eventual_encoding, formatter, iterator)

**Description:** Render this `Tag` and its contents as a Unicode string.

:param indent_level: Each line of the rendering will be
   indented this many levels. (The ``formatter`` decides what a
   'level' means, in terms of spaces or other characters
   output.) This is used internally in recursive calls while
   pretty-printing.
:param encoding: The encoding you intend to use when
   converting the string to a bytestring. decode() is *not*
   responsible for performing that encoding. This information
   is needed so that a real encoding can be substituted in if
   the document contains an encoding declaration (e.g. in a
   <meta> tag).
:param formatter: Either a `Formatter` object, or a string
    naming one of the standard formatters.
:param iterator: The iterator to use when navigating over the
    parse tree. This is only used by `Tag.decode_contents` and
    you probably won't need to use it.

## Class: _TreeTraversalEvent

**Description:** An internal class representing an event in the process
of traversing a parse tree.

:meta private:

### Function: _event_stream(self, iterator)

**Description:** Yield a sequence of events that can be used to reconstruct the DOM
for this element.

This lets us recreate the nested structure of this element
(e.g. when formatting it as a string) without using recursive
method calls.

This is similar in concept to the SAX API, but it's a simpler
interface designed for internal use. The events are different
from SAX and the arguments associated with the events are Tags
and other Beautiful Soup objects.

:param iterator: An alternate iterator to use when traversing
 the tree.

### Function: _indent_string(self, s, indent_level, formatter, indent_before, indent_after)

**Description:** Add indentation whitespace before and/or after a string.

:param s: The string to amend with whitespace.
:param indent_level: The indentation level; affects how much
   whitespace goes before the string.
:param indent_before: Whether or not to add whitespace
   before the string.
:param indent_after: Whether or not to add whitespace
   (a newline) after the string.

### Function: _format_tag(self, eventual_encoding, formatter, opening)

### Function: _should_pretty_print(self, indent_level)

**Description:** Should this tag be pretty-printed?

Most of them should, but some (such as <pre> in HTML
documents) should not.

### Function: prettify(self, encoding, formatter)

### Function: prettify(self, encoding, formatter)

### Function: prettify(self, encoding, formatter)

**Description:** Pretty-print this `Tag` as a string or bytestring.

:param encoding: The encoding of the bytestring, or None if you want Unicode.
:param formatter: A Formatter object, or a string naming one of
    the standard formatters.
:return: A string (if no ``encoding`` is provided) or a bytestring
    (otherwise).

### Function: decode_contents(self, indent_level, eventual_encoding, formatter)

**Description:** Renders the contents of this tag as a Unicode string.

:param indent_level: Each line of the rendering will be
   indented this many levels. (The formatter decides what a
   'level' means in terms of spaces or other characters
   output.) Used internally in recursive calls while
   pretty-printing.

:param eventual_encoding: The tag is destined to be
   encoded into this encoding. decode_contents() is *not*
   responsible for performing that encoding. This information
   is needed so that a real encoding can be substituted in if
   the document contains an encoding declaration (e.g. in a
   <meta> tag).

:param formatter: A `Formatter` object, or a string naming one of
    the standard Formatters.

### Function: encode_contents(self, indent_level, encoding, formatter)

**Description:** Renders the contents of this PageElement as a bytestring.

:param indent_level: Each line of the rendering will be
   indented this many levels. (The ``formatter`` decides what a
   'level' means, in terms of spaces or other characters
   output.) This is used internally in recursive calls while
   pretty-printing.
:param formatter: Either a `Formatter` object, or a string naming one of
    the standard formatters.
:param encoding: The bytestring will be in this encoding.

### Function: renderContents(self, encoding, prettyPrint, indentLevel)

**Description:** Deprecated method for BS3 compatibility.

:meta private:

### Function: find(self, name, attrs, recursive, string)

### Function: find(self, name, attrs, recursive, string)

### Function: find(self, name, attrs, recursive, string)

**Description:** Look in the children of this PageElement and find the first
PageElement that matches the given criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param recursive: If this is True, find() will perform a
    recursive search of this Tag's children. Otherwise,
    only the direct children will be considered.
:param string: A filter on the `Tag.string` attribute.
:kwargs: Additional filters on attribute values.

### Function: find_all(self, name, attrs, recursive, string, limit, _stacklevel)

### Function: find_all(self, name, attrs, recursive, string, limit, _stacklevel)

### Function: find_all(self, name, attrs, recursive, string, limit, _stacklevel)

**Description:** Look in the children of this `PageElement` and find all
`PageElement` objects that match the given criteria.

All find_* methods take a common set of arguments. See the online
documentation for detailed explanations.

:param name: A filter on tag name.
:param attrs: Additional filters on attribute values.
:param recursive: If this is True, find_all() will perform a
    recursive search of this PageElement's children. Otherwise,
    only the direct children will be considered.
:param limit: Stop looking after finding this many results.
:param _stacklevel: Used internally to improve warning messages.
:kwargs: Additional filters on attribute values.

### Function: children(self)

**Description:** Iterate over all direct children of this `PageElement`.

### Function: self_and_descendants(self)

**Description:** Iterate over this `Tag` and its children in a
breadth-first sequence.

### Function: descendants(self)

**Description:** Iterate over all children of this `Tag` in a
breadth-first sequence.

### Function: select_one(self, selector, namespaces)

**Description:** Perform a CSS selection operation on the current element.

:param selector: A CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
   used in the CSS selector to namespace URIs. By default,
   Beautiful Soup will use the prefixes it encountered while
   parsing the document.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   soupsieve.select() method.

### Function: select(self, selector, namespaces, limit)

**Description:** Perform a CSS selection operation on the current element.

This uses the SoupSieve library.

:param selector: A string containing a CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
   used in the CSS selector to namespace URIs. By default,
   Beautiful Soup will use the prefixes it encountered while
   parsing the document.

:param limit: After finding this number of results, stop looking.

:param kwargs: Keyword arguments to be passed into SoupSieve's
   soupsieve.select() method.

### Function: css(self)

**Description:** Return an interface to the CSS selector API.

### Function: childGenerator(self)

**Description:** Deprecated generator.

:meta private:

### Function: recursiveChildGenerator(self)

**Description:** Deprecated generator.

:meta private:

### Function: has_key(self, key)

**Description:** Deprecated method. This was kind of misleading because has_key()
(attributes) was different from __in__ (contents).

has_key() is gone in Python 3, anyway.

:meta private:

### Function: __init__(self, source, result)

### Function: __getattr__(self, key)

**Description:** Raise a helpful exception to explain a common code fix.

### Function: rewrite(match)
