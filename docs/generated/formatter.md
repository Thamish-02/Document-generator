## AI Summary

A file named formatter.py.


## Class: Formatter

**Description:** Describes a strategy to use when outputting a parse tree to a string.

Some parts of this strategy come from the distinction between
HTML4, HTML5, and XML. Others are configurable by the user.

Formatters are passed in as the `formatter` argument to methods
like `bs4.element.Tag.encode`. Most people won't need to
think about formatters, and most people who need to think about
them can pass in one of these predefined strings as `formatter`
rather than making a new Formatter object:

For HTML documents:
 * 'html' - HTML entity substitution for generic HTML documents. (default)
 * 'html5' - HTML entity substitution for HTML5 documents, as
             well as some optimizations in the way tags are rendered.
 * 'html5-4.12.0' - The version of the 'html5' formatter used prior to
                    Beautiful Soup 4.13.0.
 * 'minimal' - Only make the substitutions necessary to guarantee
               valid HTML.
 * None - Do not perform any substitution. This will be faster
          but may result in invalid markup.

For XML documents:
 * 'html' - Entity substitution for XHTML documents.
 * 'minimal' - Only make the substitutions necessary to guarantee
               valid XML. (default)
 * None - Do not perform any substitution. This will be faster
          but may result in invalid markup.

## Class: HTMLFormatter

**Description:** A generic Formatter for HTML.

## Class: XMLFormatter

**Description:** A generic Formatter for XML.

### Function: _default(self, language, value, kwarg)

### Function: __init__(self, language, entity_substitution, void_element_close_prefix, cdata_containing_tags, empty_attributes_are_booleans, indent)

**Description:** Constructor.

:param language: This should be `Formatter.XML` if you are formatting
   XML markup and `Formatter.HTML` if you are formatting HTML markup.

:param entity_substitution: A function to call to replace special
   characters with XML/HTML entities. For examples, see
   bs4.dammit.EntitySubstitution.substitute_html and substitute_xml.
:param void_element_close_prefix: By default, void elements
   are represented as <tag/> (XML rules) rather than <tag>
   (HTML rules). To get <tag>, pass in the empty string.
:param cdata_containing_tags: The set of tags that are defined
   as containing CDATA in this dialect. For example, in HTML,
   <script> and <style> tags are defined as containing CDATA,
   and their contents should not be formatted.
:param empty_attributes_are_booleans: If this is set to true,
  then attributes whose values are sent to the empty string
  will be treated as `HTML boolean
  attributes<https://dev.w3.org/html5/spec-LC/common-microsyntaxes.html#boolean-attributes>`_. (Attributes
  whose value is None are always rendered this way.)
:param indent: If indent is a non-negative integer or string,
    then the contents of elements will be indented
    appropriately when pretty-printing. An indent level of 0,
    negative, or "" will only insert newlines. Using a
    positive integer indent indents that many spaces per
    level. If indent is a string (such as "\t"), that string
    is used to indent each level. The default behavior is to
    indent one space per level.

### Function: substitute(self, ns)

**Description:** Process a string that needs to undergo entity substitution.
This may be a string encountered in an attribute value or as
text.

:param ns: A string.
:return: The same string but with certain characters replaced by named
   or numeric entities.

### Function: attribute_value(self, value)

**Description:** Process the value of an attribute.

:param ns: A string.
:return: A string with certain characters replaced by named
   or numeric entities.

### Function: attributes(self, tag)

**Description:** Reorder a tag's attributes however you want.

By default, attributes are sorted alphabetically. This makes
behavior consistent between Python 2 and Python 3, and preserves
backwards compatibility with older versions of Beautiful Soup.

If `empty_attributes_are_booleans` is True, then
attributes whose values are set to the empty string will be
treated as boolean attributes.

### Function: __init__(self, entity_substitution, void_element_close_prefix, cdata_containing_tags, empty_attributes_are_booleans, indent)

### Function: __init__(self, entity_substitution, void_element_close_prefix, cdata_containing_tags, empty_attributes_are_booleans, indent)
