## AI Summary

A file named pofile.py.


### Function: unescape(string)

**Description:** Reverse `escape` the given string.

>>> print(unescape('"Say:\\n  \\"hello, world!\\"\\n"'))
Say:
  "hello, world!"
<BLANKLINE>

:param string: the string to unescape

### Function: denormalize(string)

**Description:** Reverse the normalization done by the `normalize` function.

>>> print(denormalize(r'''""
... "Say:\n"
... "  \"hello, world!\"\n"'''))
Say:
  "hello, world!"
<BLANKLINE>

>>> print(denormalize(r'''""
... "Say:\n"
... "  \"Lorem ipsum dolor sit "
... "amet, consectetur adipisicing"
... " elit, \"\n"'''))
Say:
  "Lorem ipsum dolor sit amet, consectetur adipisicing elit, "
<BLANKLINE>

:param string: the string to denormalize

### Function: _extract_locations(line)

**Description:** Extract locations from location comments.

Locations are extracted while properly handling First Strong
Isolate (U+2068) and Pop Directional Isolate (U+2069), used by
gettext to enclose filenames with spaces and tabs in their names.

## Class: PoFileError

**Description:** Exception thrown by PoParser when an invalid po file is encountered.

## Class: _NormalizedString

## Class: PoFileParser

**Description:** Support class to  read messages from a ``gettext`` PO (portable object) file
and add them to a `Catalog`

See `read_po` for simple cases.

### Function: read_po(fileobj, locale, domain, ignore_obsolete, charset, abort_invalid)

**Description:** Read messages from a ``gettext`` PO (portable object) file from the given
file-like object (or an iterable of lines) and return a `Catalog`.

>>> from datetime import datetime
>>> from io import StringIO
>>> buf = StringIO('''
... #: main.py:1
... #, fuzzy, python-format
... msgid "foo %(name)s"
... msgstr "quux %(name)s"
...
... # A user comment
... #. An auto comment
... #: main.py:3
... msgid "bar"
... msgid_plural "baz"
... msgstr[0] "bar"
... msgstr[1] "baaz"
... ''')
>>> catalog = read_po(buf)
>>> catalog.revision_date = datetime(2007, 4, 1)

>>> for message in catalog:
...     if message.id:
...         print((message.id, message.string))
...         print(' ', (message.locations, sorted(list(message.flags))))
...         print(' ', (message.user_comments, message.auto_comments))
(u'foo %(name)s', u'quux %(name)s')
  ([(u'main.py', 1)], [u'fuzzy', u'python-format'])
  ([], [])
((u'bar', u'baz'), (u'bar', u'baaz'))
  ([(u'main.py', 3)], [])
  ([u'A user comment'], [u'An auto comment'])

.. versionadded:: 1.0
   Added support for explicit charset argument.

:param fileobj: the file-like object (or iterable of lines) to read the PO file from
:param locale: the locale identifier or `Locale` object, or `None`
               if the catalog is not bound to a locale (which basically
               means it's a template)
:param domain: the message domain
:param ignore_obsolete: whether to ignore obsolete messages in the input
:param charset: the character set of the catalog.
:param abort_invalid: abort read if po file is invalid

### Function: escape(string)

**Description:** Escape the given string so that it can be included in double-quoted
strings in ``PO`` files.

>>> escape('''Say:
...   "hello, world!"
... ''')
'"Say:\\n  \\"hello, world!\\"\\n"'

:param string: the string to escape

### Function: normalize(string, prefix, width)

**Description:** Convert a string into a format that is appropriate for .po files.

>>> print(normalize('''Say:
...   "hello, world!"
... ''', width=None))
""
"Say:\n"
"  \"hello, world!\"\n"

>>> print(normalize('''Say:
...   "Lorem ipsum dolor sit amet, consectetur adipisicing elit, "
... ''', width=32))
""
"Say:\n"
"  \"Lorem ipsum dolor sit "
"amet, consectetur adipisicing"
" elit, \"\n"

:param string: the string to normalize
:param prefix: a string that should be prepended to every line
:param width: the maximum line width; use `None`, 0, or a negative number
              to completely disable line wrapping

### Function: _enclose_filename_if_necessary(filename)

**Description:** Enclose filenames which include white spaces or tabs.

Do the same as gettext and enclose filenames which contain white
spaces or tabs with First Strong Isolate (U+2068) and Pop
Directional Isolate (U+2069).

### Function: write_po(fileobj, catalog, width, no_location, omit_header, sort_output, sort_by_file, ignore_obsolete, include_previous, include_lineno)

**Description:** Write a ``gettext`` PO (portable object) template file for a given
message catalog to the provided file-like object.

>>> catalog = Catalog()
>>> catalog.add(u'foo %(name)s', locations=[('main.py', 1)],
...             flags=('fuzzy',))
<Message...>
>>> catalog.add((u'bar', u'baz'), locations=[('main.py', 3)])
<Message...>
>>> from io import BytesIO
>>> buf = BytesIO()
>>> write_po(buf, catalog, omit_header=True)
>>> print(buf.getvalue().decode("utf8"))
#: main.py:1
#, fuzzy, python-format
msgid "foo %(name)s"
msgstr ""
<BLANKLINE>
#: main.py:3
msgid "bar"
msgid_plural "baz"
msgstr[0] ""
msgstr[1] ""
<BLANKLINE>
<BLANKLINE>

:param fileobj: the file-like object to write to
:param catalog: the `Catalog` instance
:param width: the maximum line width for the generated output; use `None`,
              0, or a negative number to completely disable line wrapping
:param no_location: do not emit a location comment for every message
:param omit_header: do not include the ``msgid ""`` entry at the top of the
                    output
:param sort_output: whether to sort the messages in the output by msgid
:param sort_by_file: whether to sort the messages in the output by their
                     locations
:param ignore_obsolete: whether to ignore obsolete messages and not include
                        them in the output; by default they are included as
                        comments
:param include_previous: include the old msgid as a comment when
                         updating the catalog
:param include_lineno: include line number in the location comment

### Function: generate_po(catalog)

**Description:** Yield text strings representing a ``gettext`` PO (portable object) file.

See `write_po()` for a more detailed description.

### Function: _sort_messages(messages, sort_by)

**Description:** Sort the given message iterable by the given criteria.

Always returns a list.

:param messages: An iterable of Messages.
:param sort_by: Sort by which criteria? Options are `message` and `location`.
:return: list[Message]

### Function: replace_escapes(match)

### Function: __init__(self, message, catalog, line, lineno)

### Function: __init__(self)

### Function: append(self, s)

### Function: denormalize(self)

### Function: __bool__(self)

### Function: __repr__(self)

### Function: __cmp__(self, other)

### Function: __gt__(self, other)

### Function: __lt__(self, other)

### Function: __ge__(self, other)

### Function: __le__(self, other)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __init__(self, catalog, ignore_obsolete, abort_invalid)

### Function: _reset_message_state(self)

### Function: _add_message(self)

**Description:** Add a message to the catalog based on the current parser state and
clear the state ready to process the next message.

### Function: _finish_current_message(self)

### Function: _process_message_line(self, lineno, line, obsolete)

### Function: _process_keyword_line(self, lineno, line, obsolete)

### Function: _process_string_continuation_line(self, line, lineno)

### Function: _process_comment(self, line)

### Function: parse(self, fileobj)

**Description:** Reads from the file-like object `fileobj` and adds any po file
units found in it to the `Catalog` supplied to the constructor.

### Function: _invalid_pofile(self, line, lineno, msg)

### Function: _format_comment(comment, prefix)

### Function: _format_message(message, prefix)
