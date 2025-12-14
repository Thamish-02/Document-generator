## AI Summary

A file named support.py.


## Class: Format

**Description:** Wrapper class providing the various date and number formatting functions
bound to a specific locale and time-zone.

>>> from babel.util import UTC
>>> from datetime import date
>>> fmt = Format('en_US', UTC)
>>> fmt.date(date(2007, 4, 1))
u'Apr 1, 2007'
>>> fmt.decimal(1.2345)
u'1.234'

## Class: LazyProxy

**Description:** Class for proxy objects that delegate to a specified function to evaluate
the actual object.

>>> def greeting(name='world'):
...     return 'Hello, %s!' % name
>>> lazy_greeting = LazyProxy(greeting, name='Joe')
>>> print(lazy_greeting)
Hello, Joe!
>>> u'  ' + lazy_greeting
u'  Hello, Joe!'
>>> u'(%s)' % lazy_greeting
u'(Hello, Joe!)'

This can be used, for example, to implement lazy translation functions that
delay the actual translation until the string is actually used. The
rationale for such behavior is that the locale of the user may not always
be available. In web applications, you only know the locale when processing
a request.

The proxy implementation attempts to be as complete as possible, so that
the lazy objects should mostly work as expected, for example for sorting:

>>> greetings = [
...     LazyProxy(greeting, 'world'),
...     LazyProxy(greeting, 'Joe'),
...     LazyProxy(greeting, 'universe'),
... ]
>>> greetings.sort()
>>> for greeting in greetings:
...     print(greeting)
Hello, Joe!
Hello, universe!
Hello, world!

## Class: NullTranslations

## Class: Translations

**Description:** An extended translation catalog class.

### Function: _locales_to_names(locales)

**Description:** Normalize a `locales` argument to a list of locale names.

:param locales: the list of locales in order of preference (items in
                this list can be either `Locale` objects or locale
                strings)

### Function: __init__(self, locale, tzinfo)

**Description:** Initialize the formatter.

:param locale: the locale identifier or `Locale` instance
:param tzinfo: the time-zone info (a `tzinfo` instance or `None`)
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.

### Function: date(self, date, format)

**Description:** Return a date formatted according to the given pattern.

>>> from datetime import date
>>> fmt = Format('en_US')
>>> fmt.date(date(2007, 4, 1))
u'Apr 1, 2007'

### Function: datetime(self, datetime, format)

**Description:** Return a date and time formatted according to the given pattern.

>>> from datetime import datetime
>>> from babel.dates import get_timezone
>>> fmt = Format('en_US', tzinfo=get_timezone('US/Eastern'))
>>> fmt.datetime(datetime(2007, 4, 1, 15, 30))
u'Apr 1, 2007, 11:30:00 AM'

### Function: time(self, time, format)

**Description:** Return a time formatted according to the given pattern.

>>> from datetime import datetime
>>> from babel.dates import get_timezone
>>> fmt = Format('en_US', tzinfo=get_timezone('US/Eastern'))
>>> fmt.time(datetime(2007, 4, 1, 15, 30))
u'11:30:00 AM'

### Function: timedelta(self, delta, granularity, threshold, format, add_direction)

**Description:** Return a time delta according to the rules of the given locale.

>>> from datetime import timedelta
>>> fmt = Format('en_US')
>>> fmt.timedelta(timedelta(weeks=11))
u'3 months'

### Function: number(self, number)

**Description:** Return an integer number formatted for the locale.

>>> fmt = Format('en_US')
>>> fmt.number(1099)
u'1,099'

### Function: decimal(self, number, format)

**Description:** Return a decimal number formatted for the locale.

>>> fmt = Format('en_US')
>>> fmt.decimal(1.2345)
u'1.234'

### Function: compact_decimal(self, number, format_type, fraction_digits)

**Description:** Return a number formatted in compact form for the locale.

>>> fmt = Format('en_US')
>>> fmt.compact_decimal(123456789)
u'123M'
>>> fmt.compact_decimal(1234567, format_type='long', fraction_digits=2)
'1.23 million'

### Function: currency(self, number, currency)

**Description:** Return a number in the given currency formatted for the locale.
        

### Function: compact_currency(self, number, currency, format_type, fraction_digits)

**Description:** Return a number in the given currency formatted for the locale
using the compact number format.

>>> Format('en_US').compact_currency(1234567, "USD", format_type='short', fraction_digits=2)
'$1.23M'

### Function: percent(self, number, format)

**Description:** Return a number formatted as percentage for the locale.

>>> fmt = Format('en_US')
>>> fmt.percent(0.34)
u'34%'

### Function: scientific(self, number)

**Description:** Return a number formatted using scientific notation for the locale.
        

### Function: __init__(self, func)

### Function: value(self)

### Function: __contains__(self, key)

### Function: __bool__(self)

### Function: __dir__(self)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __str__(self)

### Function: __add__(self, other)

### Function: __radd__(self, other)

### Function: __mod__(self, other)

### Function: __rmod__(self, other)

### Function: __mul__(self, other)

### Function: __rmul__(self, other)

### Function: __call__(self)

### Function: __lt__(self, other)

### Function: __le__(self, other)

### Function: __eq__(self, other)

### Function: __ne__(self, other)

### Function: __gt__(self, other)

### Function: __ge__(self, other)

### Function: __delattr__(self, name)

### Function: __getattr__(self, name)

### Function: __setattr__(self, name, value)

### Function: __delitem__(self, key)

### Function: __getitem__(self, key)

### Function: __setitem__(self, key, value)

### Function: __copy__(self)

### Function: __deepcopy__(self, memo)

### Function: __init__(self, fp)

**Description:** Initialize a simple translations class which is not backed by a
real catalog. Behaves similar to gettext.NullTranslations but also
offers Babel's on *gettext methods (e.g. 'dgettext()').

:param fp: a file-like object (ignored in this class)

### Function: dgettext(self, domain, message)

**Description:** Like ``gettext()``, but look the message up in the specified
domain.

### Function: ldgettext(self, domain, message)

**Description:** Like ``lgettext()``, but look the message up in the specified
domain.

### Function: udgettext(self, domain, message)

**Description:** Like ``ugettext()``, but look the message up in the specified
domain.

### Function: dngettext(self, domain, singular, plural, num)

**Description:** Like ``ngettext()``, but look the message up in the specified
domain.

### Function: ldngettext(self, domain, singular, plural, num)

**Description:** Like ``lngettext()``, but look the message up in the specified
domain.

### Function: udngettext(self, domain, singular, plural, num)

**Description:** Like ``ungettext()`` but look the message up in the specified
domain.

### Function: pgettext(self, context, message)

**Description:** Look up the `context` and `message` id in the catalog and return the
corresponding message string, as an 8-bit string encoded with the
catalog's charset encoding, if known.  If there is no entry in the
catalog for the `message` id and `context` , and a fallback has been
set, the look up is forwarded to the fallback's ``pgettext()``
method. Otherwise, the `message` id is returned.

### Function: lpgettext(self, context, message)

**Description:** Equivalent to ``pgettext()``, but the translation is returned in the
preferred system encoding, if no other encoding was explicitly set with
``bind_textdomain_codeset()``.

### Function: npgettext(self, context, singular, plural, num)

**Description:** Do a plural-forms lookup of a message id.  `singular` is used as the
message id for purposes of lookup in the catalog, while `num` is used to
determine which plural form to use.  The returned message string is an
8-bit string encoded with the catalog's charset encoding, if known.

If the message id for `context` is not found in the catalog, and a
fallback is specified, the request is forwarded to the fallback's
``npgettext()`` method.  Otherwise, when ``num`` is 1 ``singular`` is
returned, and ``plural`` is returned in all other cases.

### Function: lnpgettext(self, context, singular, plural, num)

**Description:** Equivalent to ``npgettext()``, but the translation is returned in the
preferred system encoding, if no other encoding was explicitly set with
``bind_textdomain_codeset()``.

### Function: upgettext(self, context, message)

**Description:** Look up the `context` and `message` id in the catalog and return the
corresponding message string, as a Unicode string.  If there is no entry
in the catalog for the `message` id and `context`, and a fallback has
been set, the look up is forwarded to the fallback's ``upgettext()``
method.  Otherwise, the `message` id is returned.

### Function: unpgettext(self, context, singular, plural, num)

**Description:** Do a plural-forms lookup of a message id.  `singular` is used as the
message id for purposes of lookup in the catalog, while `num` is used to
determine which plural form to use.  The returned message string is a
Unicode string.

If the message id for `context` is not found in the catalog, and a
fallback is specified, the request is forwarded to the fallback's
``unpgettext()`` method.  Otherwise, when `num` is 1 `singular` is
returned, and `plural` is returned in all other cases.

### Function: dpgettext(self, domain, context, message)

**Description:** Like `pgettext()`, but look the message up in the specified
`domain`.

### Function: udpgettext(self, domain, context, message)

**Description:** Like `upgettext()`, but look the message up in the specified
`domain`.

### Function: ldpgettext(self, domain, context, message)

**Description:** Equivalent to ``dpgettext()``, but the translation is returned in the
preferred system encoding, if no other encoding was explicitly set with
``bind_textdomain_codeset()``.

### Function: dnpgettext(self, domain, context, singular, plural, num)

**Description:** Like ``npgettext``, but look the message up in the specified
`domain`.

### Function: udnpgettext(self, domain, context, singular, plural, num)

**Description:** Like ``unpgettext``, but look the message up in the specified
`domain`.

### Function: ldnpgettext(self, domain, context, singular, plural, num)

**Description:** Equivalent to ``dnpgettext()``, but the translation is returned in
the preferred system encoding, if no other encoding was explicitly set
with ``bind_textdomain_codeset()``.

### Function: __init__(self, fp, domain)

**Description:** Initialize the translations catalog.

:param fp: the file-like object the translation should be read from
:param domain: the message domain (default: 'messages')

### Function: load(cls, dirname, locales, domain)

**Description:** Load translations from the given directory.

:param dirname: the directory containing the ``MO`` files
:param locales: the list of locales in order of preference (items in
                this list can be either `Locale` objects or locale
                strings)
:param domain: the message domain (default: 'messages')

### Function: __repr__(self)

### Function: add(self, translations, merge)

**Description:** Add the given translations to the catalog.

If the domain of the translations is different than that of the
current catalog, they are added as a catalog that is only accessible
by the various ``d*gettext`` functions.

:param translations: the `Translations` instance with the messages to
                     add
:param merge: whether translations for message domains that have
              already been added should be merged with the existing
              translations

### Function: merge(self, translations)

**Description:** Merge the given translations into the catalog.

Message translations in the specified catalog override any messages
with the same identifier in the existing catalog.

:param translations: the `Translations` instance with the messages to
                     merge
