## AI Summary

A file named lists.py.


### Function: __getattr__(name)

### Function: format_list(lst, style, locale)

**Description:** Format the items in `lst` as a list.

>>> format_list(['apples', 'oranges', 'pears'], locale='en')
u'apples, oranges, and pears'
>>> format_list(['apples', 'oranges', 'pears'], locale='zh')
u'apples、oranges和pears'
>>> format_list(['omena', 'peruna', 'aplari'], style='or', locale='fi')
u'omena, peruna tai aplari'

Not all styles are necessarily available in all locales.
The function will attempt to fall back to replacement styles according to the rules
set forth in the CLDR root XML file, and raise a ValueError if no suitable replacement
can be found.

The following text is verbatim from the Unicode TR35-49 spec [1].

* standard:
  A typical 'and' list for arbitrary placeholders.
  eg. "January, February, and March"
* standard-short:
  A short version of an 'and' list, suitable for use with short or abbreviated placeholder values.
  eg. "Jan., Feb., and Mar."
* or:
  A typical 'or' list for arbitrary placeholders.
  eg. "January, February, or March"
* or-short:
  A short version of an 'or' list.
  eg. "Jan., Feb., or Mar."
* unit:
  A list suitable for wide units.
  eg. "3 feet, 7 inches"
* unit-short:
  A list suitable for short units
  eg. "3 ft, 7 in"
* unit-narrow:
  A list suitable for narrow units, where space on the screen is very limited.
  eg. "3′ 7″"

[1]: https://www.unicode.org/reports/tr35/tr35-49/tr35-general.html#ListPatterns

:param lst: a sequence of items to format in to a list
:param style: the style to format the list with. See above for description.
:param locale: the locale. Defaults to the system locale.

### Function: _resolve_list_style(locale, style)
