## AI Summary

A file named catalog.py.


### Function: get_close_matches(word, possibilities, n, cutoff)

**Description:** A modified version of ``difflib.get_close_matches``.

It just passes ``autojunk=False`` to the ``SequenceMatcher``, to work
around https://github.com/python/cpython/issues/90825.

### Function: _has_python_brace_format(string)

### Function: _parse_datetime_header(value)

## Class: Message

**Description:** Representation of a single message in a catalog.

## Class: TranslationError

**Description:** Exception thrown by translation checkers when invalid message
translations are encountered.

### Function: parse_separated_header(value)

### Function: _force_text(s, encoding, errors)

## Class: Catalog

**Description:** Representation of a message catalog.

### Function: __init__(self, id, string, locations, flags, auto_comments, user_comments, previous_id, lineno, context)

**Description:** Create the message object.

:param id: the message ID, or a ``(singular, plural)`` tuple for
           pluralizable messages
:param string: the translated message string, or a
               ``(singular, plural)`` tuple for pluralizable messages
:param locations: a sequence of ``(filename, lineno)`` tuples
:param flags: a set or sequence of flags
:param auto_comments: a sequence of automatic comments for the message
:param user_comments: a sequence of user comments for the message
:param previous_id: the previous message ID, or a ``(singular, plural)``
                    tuple for pluralizable messages
:param lineno: the line number on which the msgid line was found in the
               PO file, if any
:param context: the message context

### Function: __repr__(self)

### Function: __cmp__(self, other)

**Description:** Compare Messages, taking into account plural ids

### Function: __gt__(self, other)

### Function: __lt__(self, other)

### Function: __ge__(self, other)

### Function: __le__(self, other)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: is_identical(self, other)

**Description:** Checks whether messages are identical, taking into account all
properties.

### Function: clone(self)

### Function: check(self, catalog)

**Description:** Run various validation checks on the message.  Some validations
are only performed if the catalog is provided.  This method returns
a sequence of `TranslationError` objects.

:rtype: ``iterator``
:param catalog: A catalog instance that is passed to the checkers
:see: `Catalog.check` for a way to perform checks for all messages
      in a catalog.

### Function: fuzzy(self)

**Description:** Whether the translation is fuzzy.

>>> Message('foo').fuzzy
False
>>> msg = Message('foo', 'foo', flags=['fuzzy'])
>>> msg.fuzzy
True
>>> msg
<Message 'foo' (flags: ['fuzzy'])>

:type:  `bool`

### Function: pluralizable(self)

**Description:** Whether the message is plurizable.

>>> Message('foo').pluralizable
False
>>> Message(('foo', 'bar')).pluralizable
True

:type:  `bool`

### Function: python_format(self)

**Description:** Whether the message contains Python-style parameters.

>>> Message('foo %(name)s bar').python_format
True
>>> Message(('foo %(name)s', 'foo %(name)s')).python_format
True

:type:  `bool`

### Function: python_brace_format(self)

**Description:** Whether the message contains Python f-string parameters.

>>> Message('Hello, {name}!').python_brace_format
True
>>> Message(('One apple', '{count} apples')).python_brace_format
True

:type:  `bool`

### Function: __init__(self, locale, domain, header_comment, project, version, copyright_holder, msgid_bugs_address, creation_date, revision_date, last_translator, language_team, charset, fuzzy)

**Description:** Initialize the catalog object.

:param locale: the locale identifier or `Locale` object, or `None`
               if the catalog is not bound to a locale (which basically
               means it's a template)
:param domain: the message domain
:param header_comment: the header comment as string, or `None` for the
                       default header
:param project: the project's name
:param version: the project's version
:param copyright_holder: the copyright holder of the catalog
:param msgid_bugs_address: the email address or URL to submit bug
                           reports to
:param creation_date: the date the catalog was created
:param revision_date: the date the catalog was revised
:param last_translator: the name and email of the last translator
:param language_team: the name and email of the language team
:param charset: the encoding to use in the output (defaults to utf-8)
:param fuzzy: the fuzzy bit on the catalog header

### Function: _set_locale(self, locale)

### Function: _get_locale(self)

### Function: _get_locale_identifier(self)

### Function: _get_header_comment(self)

### Function: _set_header_comment(self, string)

### Function: _get_mime_headers(self)

### Function: _set_mime_headers(self, headers)

### Function: num_plurals(self)

**Description:** The number of plurals used by the catalog or locale.

>>> Catalog(locale='en').num_plurals
2
>>> Catalog(locale='ga').num_plurals
5

:type: `int`

### Function: plural_expr(self)

**Description:** The plural expression used by the catalog or locale.

>>> Catalog(locale='en').plural_expr
'(n != 1)'
>>> Catalog(locale='ga').plural_expr
'(n==1 ? 0 : n==2 ? 1 : n>=3 && n<=6 ? 2 : n>=7 && n<=10 ? 3 : 4)'
>>> Catalog(locale='ding').plural_expr  # unknown locale
'(n != 1)'

:type: `str`

### Function: plural_forms(self)

**Description:** Return the plural forms declaration for the locale.

>>> Catalog(locale='en').plural_forms
'nplurals=2; plural=(n != 1);'
>>> Catalog(locale='pt_BR').plural_forms
'nplurals=2; plural=(n > 1);'

:type: `str`

### Function: __contains__(self, id)

**Description:** Return whether the catalog has a message with the specified ID.

### Function: __len__(self)

**Description:** The number of messages in the catalog.

This does not include the special ``msgid ""`` entry.

### Function: __iter__(self)

**Description:** Iterates through all the entries in the catalog, in the order they
were added, yielding a `Message` object for every entry.

:rtype: ``iterator``

### Function: __repr__(self)

### Function: __delitem__(self, id)

**Description:** Delete the message with the specified ID.

### Function: __getitem__(self, id)

**Description:** Return the message with the specified ID.

:param id: the message ID

### Function: __setitem__(self, id, message)

**Description:** Add or update the message with the specified ID.

>>> catalog = Catalog()
>>> catalog[u'foo'] = Message(u'foo')
>>> catalog[u'foo']
<Message u'foo' (flags: [])>

If a message with that ID is already in the catalog, it is updated
to include the locations and flags of the new message.

>>> catalog = Catalog()
>>> catalog[u'foo'] = Message(u'foo', locations=[('main.py', 1)])
>>> catalog[u'foo'].locations
[('main.py', 1)]
>>> catalog[u'foo'] = Message(u'foo', locations=[('utils.py', 5)])
>>> catalog[u'foo'].locations
[('main.py', 1), ('utils.py', 5)]

:param id: the message ID
:param message: the `Message` object

### Function: add(self, id, string, locations, flags, auto_comments, user_comments, previous_id, lineno, context)

**Description:** Add or update the message with the specified ID.

>>> catalog = Catalog()
>>> catalog.add(u'foo')
<Message ...>
>>> catalog[u'foo']
<Message u'foo' (flags: [])>

This method simply constructs a `Message` object with the given
arguments and invokes `__setitem__` with that object.

:param id: the message ID, or a ``(singular, plural)`` tuple for
           pluralizable messages
:param string: the translated message string, or a
               ``(singular, plural)`` tuple for pluralizable messages
:param locations: a sequence of ``(filename, lineno)`` tuples
:param flags: a set or sequence of flags
:param auto_comments: a sequence of automatic comments
:param user_comments: a sequence of user comments
:param previous_id: the previous message ID, or a ``(singular, plural)``
                    tuple for pluralizable messages
:param lineno: the line number on which the msgid line was found in the
               PO file, if any
:param context: the message context

### Function: check(self)

**Description:** Run various validation checks on the translations in the catalog.

For every message which fails validation, this method yield a
``(message, errors)`` tuple, where ``message`` is the `Message` object
and ``errors`` is a sequence of `TranslationError` objects.

:rtype: ``generator`` of ``(message, errors)``

### Function: get(self, id, context)

**Description:** Return the message with the specified ID and context.

:param id: the message ID
:param context: the message context, or ``None`` for no context

### Function: delete(self, id, context)

**Description:** Delete the message with the specified ID and context.

:param id: the message ID
:param context: the message context, or ``None`` for no context

### Function: update(self, template, no_fuzzy_matching, update_header_comment, keep_user_comments, update_creation_date)

**Description:** Update the catalog based on the given template catalog.

>>> from babel.messages import Catalog
>>> template = Catalog()
>>> template.add('green', locations=[('main.py', 99)])
<Message ...>
>>> template.add('blue', locations=[('main.py', 100)])
<Message ...>
>>> template.add(('salad', 'salads'), locations=[('util.py', 42)])
<Message ...>
>>> catalog = Catalog(locale='de_DE')
>>> catalog.add('blue', u'blau', locations=[('main.py', 98)])
<Message ...>
>>> catalog.add('head', u'Kopf', locations=[('util.py', 33)])
<Message ...>
>>> catalog.add(('salad', 'salads'), (u'Salat', u'Salate'),
...             locations=[('util.py', 38)])
<Message ...>

>>> catalog.update(template)
>>> len(catalog)
3

>>> msg1 = catalog['green']
>>> msg1.string
>>> msg1.locations
[('main.py', 99)]

>>> msg2 = catalog['blue']
>>> msg2.string
u'blau'
>>> msg2.locations
[('main.py', 100)]

>>> msg3 = catalog['salad']
>>> msg3.string
(u'Salat', u'Salate')
>>> msg3.locations
[('util.py', 42)]

Messages that are in the catalog but not in the template are removed
from the main collection, but can still be accessed via the `obsolete`
member:

>>> 'head' in catalog
False
>>> list(catalog.obsolete.values())
[<Message 'head' (flags: [])>]

:param template: the reference catalog, usually read from a POT file
:param no_fuzzy_matching: whether to use fuzzy matching of message IDs
:param update_header_comment: whether to copy the header comment from the template
:param keep_user_comments: whether to keep user comments from the old catalog
:param update_creation_date: whether to copy the creation date from the template

### Function: _to_fuzzy_match_key(self, key)

**Description:** Converts a message key to a string suitable for fuzzy matching.

### Function: _key_for(self, id, context)

**Description:** The key for a message is just the singular ID even for pluralizable
messages, but is a ``(msgid, msgctxt)`` tuple for context-specific
messages.

### Function: is_identical(self, other)

**Description:** Checks if catalogs are identical, taking into account messages and
headers.

### Function: values_to_compare(obj)

### Function: _merge(message, oldkey, newkey)
