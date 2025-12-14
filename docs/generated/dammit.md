## AI Summary

A file named dammit.py.


### Function: _chardet_dammit(s)

**Description:** Try as hard as possible to detect the encoding of a bytestring.

## Class: EntitySubstitution

**Description:** The ability to substitute XML or HTML entities for certain characters.

## Class: EncodingDetector

**Description:** This class is capable of guessing a number of possible encodings
for a bytestring.

Order of precedence:

1. Encodings you specifically tell EncodingDetector to try first
   (the ``known_definite_encodings`` argument to the constructor).

2. An encoding determined by sniffing the document's byte-order mark.

3. Encodings you specifically tell EncodingDetector to try if
   byte-order mark sniffing fails (the ``user_encodings`` argument to the
   constructor).

4. An encoding declared within the bytestring itself, either in an
   XML declaration (if the bytestring is to be interpreted as an XML
   document), or in a <meta> tag (if the bytestring is to be
   interpreted as an HTML document.)

5. An encoding detected through textual analysis by chardet,
   cchardet, or a similar external library.

6. UTF-8.

7. Windows-1252.

:param markup: Some markup in an unknown encoding.

:param known_definite_encodings: When determining the encoding
    of ``markup``, these encodings will be tried first, in
    order. In HTML terms, this corresponds to the "known
    definite encoding" step defined in `section 13.2.3.1 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding>`_.

:param user_encodings: These encodings will be tried after the
    ``known_definite_encodings`` have been tried and failed, and
    after an attempt to sniff the encoding by looking at a
    byte order mark has failed. In HTML terms, this
    corresponds to the step "user has explicitly instructed
    the user agent to override the document's character
    encoding", defined in `section 13.2.3.2 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding>`_.

:param override_encodings: A **deprecated** alias for
    ``known_definite_encodings``. Any encodings here will be tried
    immediately after the encodings in
    ``known_definite_encodings``.

:param is_html: If True, this markup is considered to be
    HTML. Otherwise it's assumed to be XML.

:param exclude_encodings: These encodings will not be tried,
    even if they otherwise would be.

## Class: UnicodeDammit

**Description:** A class for detecting the encoding of a bytestring containing an
HTML or XML document, and decoding it to Unicode. If the source
encoding is windows-1252, `UnicodeDammit` can also replace
Microsoft smart quotes with their HTML or XML equivalents.

:param markup: HTML or XML markup in an unknown encoding.

:param known_definite_encodings: When determining the encoding
    of ``markup``, these encodings will be tried first, in
    order. In HTML terms, this corresponds to the "known
    definite encoding" step defined in `section 13.2.3.1 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding>`_.

:param user_encodings: These encodings will be tried after the
    ``known_definite_encodings`` have been tried and failed, and
    after an attempt to sniff the encoding by looking at a
    byte order mark has failed. In HTML terms, this
    corresponds to the step "user has explicitly instructed
    the user agent to override the document's character
    encoding", defined in `section 13.2.3.2 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding>`_.

:param override_encodings: A **deprecated** alias for
    ``known_definite_encodings``. Any encodings here will be tried
    immediately after the encodings in
    ``known_definite_encodings``.

:param smart_quotes_to: By default, Microsoft smart quotes will,
   like all other characters, be converted to Unicode
   characters. Setting this to ``ascii`` will convert them to ASCII
   quotes instead.  Setting it to ``xml`` will convert them to XML
   entity references, and setting it to ``html`` will convert them
   to HTML entity references.

:param is_html: If True, ``markup`` is treated as an HTML
   document. Otherwise it's treated as an XML document.

:param exclude_encodings: These encodings will not be considered,
   even if the sniffing code thinks they might make sense.

### Function: _populate_class_variables(cls)

**Description:** Initialize variables used by this class to manage the plethora of
HTML5 named entities.

This function sets the following class variables:

CHARACTER_TO_HTML_ENTITY - A mapping of Unicode strings like "⦨" to
entity names like "angmsdaa". When a single Unicode string has
multiple entity names, we try to choose the most commonly-used
name.

HTML_ENTITY_TO_CHARACTER: A mapping of entity names like "angmsdaa" to
Unicode strings like "⦨".

CHARACTER_TO_HTML_ENTITY_RE: A regular expression matching (almost) any
Unicode string that corresponds to an HTML5 named entity.

CHARACTER_TO_HTML_ENTITY_WITH_AMPERSAND_RE: A very similar
regular expression to CHARACTER_TO_HTML_ENTITY_RE, but which
also matches unescaped ampersands. This is used by the 'html'
formatted to provide backwards-compatibility, even though the HTML5
spec allows most ampersands to go unescaped.

### Function: _substitute_html_entity(cls, matchobj)

**Description:** Used with a regular expression to substitute the
appropriate HTML entity for a special character string.

### Function: _substitute_xml_entity(cls, matchobj)

**Description:** Used with a regular expression to substitute the
appropriate XML entity for a special character string.

### Function: _escape_entity_name(cls, matchobj)

### Function: _escape_unrecognized_entity_name(cls, matchobj)

### Function: quoted_attribute_value(cls, value)

**Description:** Make a value into a quoted XML attribute, possibly escaping it.

 Most strings will be quoted using double quotes.

  Bob's Bar -> "Bob's Bar"

 If a string contains double quotes, it will be quoted using
 single quotes.

  Welcome to "my bar" -> 'Welcome to "my bar"'

 If a string contains both single and double quotes, the
 double quotes will be escaped, and the string will be quoted
 using double quotes.

  Welcome to "Bob's Bar" -> Welcome to &quot;Bob's bar&quot;

:param value: The XML attribute value to quote
:return: The quoted value

### Function: substitute_xml(cls, value, make_quoted_attribute)

**Description:** Replace special XML characters with named XML entities.

The less-than sign will become &lt;, the greater-than sign
will become &gt;, and any ampersands will become &amp;. If you
want ampersands that seem to be part of an entity definition
to be left alone, use `substitute_xml_containing_entities`
instead.

:param value: A string to be substituted.

:param make_quoted_attribute: If True, then the string will be
 quoted, as befits an attribute value.

:return: A version of ``value`` with special characters replaced
 with named entities.

### Function: substitute_xml_containing_entities(cls, value, make_quoted_attribute)

**Description:** Substitute XML entities for special XML characters.

:param value: A string to be substituted. The less-than sign will
  become &lt;, the greater-than sign will become &gt;, and any
  ampersands that are not part of an entity defition will
  become &amp;.

:param make_quoted_attribute: If True, then the string will be
 quoted, as befits an attribute value.

### Function: substitute_html(cls, s)

**Description:** Replace certain Unicode characters with named HTML entities.

This differs from ``data.encode(encoding, 'xmlcharrefreplace')``
in that the goal is to make the result more readable (to those
with ASCII displays) rather than to recover from
errors. There's absolutely nothing wrong with a UTF-8 string
containg a LATIN SMALL LETTER E WITH ACUTE, but replacing that
character with "&eacute;" will make it more readable to some
people.

:param s: The string to be modified.
:return: The string with some Unicode characters replaced with
   HTML entities.

### Function: substitute_html5(cls, s)

**Description:** Replace certain Unicode characters with named HTML entities
using HTML5 rules.

Specifically, this method is much less aggressive about
escaping ampersands than substitute_html. Only ambiguous
ampersands are escaped, per the HTML5 standard:

"An ambiguous ampersand is a U+0026 AMPERSAND character (&)
that is followed by one or more ASCII alphanumerics, followed
by a U+003B SEMICOLON character (;), where these characters do
not match any of the names given in the named character
references section."

Unlike substitute_html5_raw, this method assumes HTML entities
were converted to Unicode characters on the way in, as
Beautiful Soup does. By the time Beautiful Soup does its work,
the only ambiguous ampersands that need to be escaped are the
ones that were escaped in the original markup when mentioning
HTML entities.

:param s: The string to be modified.
:return: The string with some Unicode characters replaced with
   HTML entities.

### Function: substitute_html5_raw(cls, s)

**Description:** Replace certain Unicode characters with named HTML entities
using HTML5 rules.

substitute_html5_raw is similar to substitute_html5 but it is
designed for standalone use (whereas substitute_html5 is
designed for use with Beautiful Soup).

:param s: The string to be modified.
:return: The string with some Unicode characters replaced with
   HTML entities.

### Function: __init__(self, markup, known_definite_encodings, is_html, exclude_encodings, user_encodings, override_encodings)

### Function: _usable(self, encoding, tried)

**Description:** Should we even bother to try this encoding?

:param encoding: Name of an encoding.
:param tried: Encodings that have already been tried. This
    will be modified as a side effect.

### Function: encodings(self)

**Description:** Yield a number of encodings that might work for this markup.

:yield: A sequence of strings. Each is the name of an encoding
   that *might* work to convert a bytestring into Unicode.

### Function: strip_byte_order_mark(cls, data)

**Description:** If a byte-order mark is present, strip it and return the encoding it implies.

:param data: A bytestring that may or may not begin with a
   byte-order mark.

:return: A 2-tuple (data stripped of byte-order mark, encoding implied by byte-order mark)

### Function: find_declared_encoding(cls, markup, is_html, search_entire_document)

**Description:** Given a document, tries to find an encoding declared within the
text of the document itself.

An XML encoding is declared at the beginning of the document.

An HTML encoding is declared in a <meta> tag, hopefully near the
beginning of the document.

:param markup: Some markup.
:param is_html: If True, this markup is considered to be HTML. Otherwise
    it's assumed to be XML.
:param search_entire_document: Since an encoding is supposed
    to declared near the beginning of the document, most of
    the time it's only necessary to search a few kilobytes of
    data.  Set this to True to force this method to search the
    entire document.
:return: The declared encoding, if one is found.

### Function: __init__(self, markup, known_definite_encodings, smart_quotes_to, is_html, exclude_encodings, user_encodings, override_encodings)

### Function: _sub_ms_char(self, match)

**Description:** Changes a MS smart quote character to an XML or HTML
entity, or an ASCII character.

TODO: Since this is only used to convert smart quotes, it
could be simplified, and MS_CHARS_TO_ASCII made much less
parochial.

### Function: _convert_from(self, proposed, errors)

**Description:** Attempt to convert the markup to the proposed encoding.

:param proposed: The name of a character encoding.
:param errors: An error handling strategy, used when calling `str`.
:return: The converted markup, or `None` if the proposed
   encoding/error handling strategy didn't work.

### Function: _to_unicode(self, data, encoding, errors)

**Description:** Given a bytestring and its encoding, decodes the string into Unicode.

:param encoding: The name of an encoding.
:param errors: An error handling strategy, used when calling `str`.

### Function: declared_html_encoding(self)

**Description:** If the markup is an HTML document, returns the encoding, if any,
declared *inside* the document.

### Function: find_codec(self, charset)

**Description:** Look up the Python codec corresponding to a given character set.

:param charset: The name of a character set.
:return: The name of a Python codec.

### Function: _codec(self, charset)

### Function: detwingle(cls, in_bytes, main_encoding, embedded_encoding)

**Description:** Fix characters from one encoding embedded in some other encoding.

Currently the only situation supported is Windows-1252 (or its
subset ISO-8859-1), embedded in UTF-8.

:param in_bytes: A bytestring that you suspect contains
    characters from multiple encodings. Note that this *must*
    be a bytestring. If you've already converted the document
    to Unicode, you're too late.
:param main_encoding: The primary encoding of ``in_bytes``.
:param embedded_encoding: The encoding that was used to embed characters
    in the main document.
:return: A bytestring similar to ``in_bytes``, in which
  ``embedded_encoding`` characters have been converted to
  their ``main_encoding`` equivalents.
