## AI Summary

A file named numbers.py.


## Class: UnknownCurrencyError

**Description:** Exception thrown when a currency is requested for which no data is available.
    

### Function: list_currencies(locale)

**Description:** Return a `set` of normalized currency codes.

.. versionadded:: 2.5.0

:param locale: filters returned currency codes by the provided locale.
               Expected to be a locale instance or code. If no locale is
               provided, returns the list of all currencies from all
               locales.

### Function: validate_currency(currency, locale)

**Description:** Check the currency code is recognized by Babel.

Accepts a ``locale`` parameter for fined-grained validation, working as
the one defined above in ``list_currencies()`` method.

Raises a `UnknownCurrencyError` exception if the currency is unknown to Babel.

### Function: is_currency(currency, locale)

**Description:** Returns `True` only if a currency is recognized by Babel.

This method always return a Boolean and never raise.

### Function: normalize_currency(currency, locale)

**Description:** Returns the normalized identifier of any currency code.

Accepts a ``locale`` parameter for fined-grained validation, working as
the one defined above in ``list_currencies()`` method.

Returns None if the currency is unknown to Babel.

### Function: get_currency_name(currency, count, locale)

**Description:** Return the name used by the locale for the specified currency.

>>> get_currency_name('USD', locale='en_US')
u'US Dollar'

.. versionadded:: 0.9.4

:param currency: the currency code.
:param count: the optional count.  If provided the currency name
              will be pluralized to that number if possible.
:param locale: the `Locale` object or locale identifier.
               Defaults to the system currency locale or numeric locale.

### Function: get_currency_symbol(currency, locale)

**Description:** Return the symbol used by the locale for the specified currency.

>>> get_currency_symbol('USD', locale='en_US')
u'$'

:param currency: the currency code.
:param locale: the `Locale` object or locale identifier.
               Defaults to the system currency locale or numeric locale.

### Function: get_currency_precision(currency)

**Description:** Return currency's precision.

Precision is the number of decimals found after the decimal point in the
currency's format pattern.

.. versionadded:: 2.5.0

:param currency: the currency code.

### Function: get_currency_unit_pattern(currency, count, locale)

**Description:** Return the unit pattern used for long display of a currency value
for a given locale.
This is a string containing ``{0}`` where the numeric part
should be substituted and ``{1}`` where the currency long display
name should be substituted.

>>> get_currency_unit_pattern('USD', locale='en_US', count=10)
u'{0} {1}'

.. versionadded:: 2.7.0

:param currency: the currency code.
:param count: the optional count.  If provided the unit
              pattern for that number will be returned.
:param locale: the `Locale` object or locale identifier.
               Defaults to the system currency locale or numeric locale.

### Function: get_territory_currencies(territory, start_date, end_date, tender, non_tender, include_details)

### Function: get_territory_currencies(territory, start_date, end_date, tender, non_tender, include_details)

### Function: get_territory_currencies(territory, start_date, end_date, tender, non_tender, include_details)

**Description:** Returns the list of currencies for the given territory that are valid for
the given date range.  In addition to that the currency database
distinguishes between tender and non-tender currencies.  By default only
tender currencies are returned.

The return value is a list of all currencies roughly ordered by the time
of when the currency became active.  The longer the currency is being in
use the more to the left of the list it will be.

The start date defaults to today.  If no end date is given it will be the
same as the start date.  Otherwise a range can be defined.  For instance
this can be used to find the currencies in use in Austria between 1995 and
2011:

>>> from datetime import date
>>> get_territory_currencies('AT', date(1995, 1, 1), date(2011, 1, 1))
['ATS', 'EUR']

Likewise it's also possible to find all the currencies in use on a
single date:

>>> get_territory_currencies('AT', date(1995, 1, 1))
['ATS']
>>> get_territory_currencies('AT', date(2011, 1, 1))
['EUR']

By default the return value only includes tender currencies.  This
however can be changed:

>>> get_territory_currencies('US')
['USD']
>>> get_territory_currencies('US', tender=False, non_tender=True,
...                          start_date=date(2014, 1, 1))
['USN', 'USS']

.. versionadded:: 2.0

:param territory: the name of the territory to find the currency for.
:param start_date: the start date.  If not given today is assumed.
:param end_date: the end date.  If not given the start date is assumed.
:param tender: controls whether tender currencies should be included.
:param non_tender: controls whether non-tender currencies should be
                   included.
:param include_details: if set to `True`, instead of returning currency
                        codes the return value will be dictionaries
                        with detail information.  In that case each
                        dictionary will have the keys ``'currency'``,
                        ``'from'``, ``'to'``, and ``'tender'``.

### Function: _get_numbering_system(locale, numbering_system)

### Function: _get_number_symbols(locale)

## Class: UnsupportedNumberingSystemError

**Description:** Exception thrown when an unsupported numbering system is requested for the given Locale.

### Function: get_decimal_symbol(locale)

**Description:** Return the symbol used by the locale to separate decimal fractions.

>>> get_decimal_symbol('en_US')
u'.'
>>> get_decimal_symbol('ar_EG', numbering_system='default')
u'٫'
>>> get_decimal_symbol('ar_EG', numbering_system='latn')
u'.'

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: get_plus_sign_symbol(locale)

**Description:** Return the plus sign symbol used by the current locale.

>>> get_plus_sign_symbol('en_US')
u'+'
>>> get_plus_sign_symbol('ar_EG', numbering_system='default')
u'؜+'
>>> get_plus_sign_symbol('ar_EG', numbering_system='latn')
u'‎+'

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: get_minus_sign_symbol(locale)

**Description:** Return the plus sign symbol used by the current locale.

>>> get_minus_sign_symbol('en_US')
u'-'
>>> get_minus_sign_symbol('ar_EG', numbering_system='default')
u'؜-'
>>> get_minus_sign_symbol('ar_EG', numbering_system='latn')
u'‎-'

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: get_exponential_symbol(locale)

**Description:** Return the symbol used by the locale to separate mantissa and exponent.

>>> get_exponential_symbol('en_US')
u'E'
>>> get_exponential_symbol('ar_EG', numbering_system='default')
u'أس'
>>> get_exponential_symbol('ar_EG', numbering_system='latn')
u'E'

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: get_group_symbol(locale)

**Description:** Return the symbol used by the locale to separate groups of thousands.

>>> get_group_symbol('en_US')
u','
>>> get_group_symbol('ar_EG', numbering_system='default')
u'٬'
>>> get_group_symbol('ar_EG', numbering_system='latn')
u','

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: get_infinity_symbol(locale)

**Description:** Return the symbol used by the locale to represent infinity.

>>> get_infinity_symbol('en_US')
u'∞'
>>> get_infinity_symbol('ar_EG', numbering_system='default')
u'∞'
>>> get_infinity_symbol('ar_EG', numbering_system='latn')
u'∞'

:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: format_number(number, locale)

**Description:** Return the given number formatted for a specific locale.

>>> format_number(1099, locale='en_US')  # doctest: +SKIP
u'1,099'
>>> format_number(1099, locale='de_DE')  # doctest: +SKIP
u'1.099'

.. deprecated:: 2.6.0

   Use babel.numbers.format_decimal() instead.

:param number: the number to format
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.

### Function: get_decimal_precision(number)

**Description:** Return maximum precision of a decimal instance's fractional part.

Precision is extracted from the fractional part only.

### Function: get_decimal_quantum(precision)

**Description:** Return minimal quantum of a number, as defined by precision.

### Function: format_decimal(number, format, locale, decimal_quantization, group_separator)

**Description:** Return the given decimal number formatted for a specific locale.

>>> format_decimal(1.2345, locale='en_US')
u'1.234'
>>> format_decimal(1.2346, locale='en_US')
u'1.235'
>>> format_decimal(-1.2346, locale='en_US')
u'-1.235'
>>> format_decimal(1.2345, locale='sv_SE')
u'1,234'
>>> format_decimal(1.2345, locale='de')
u'1,234'
>>> format_decimal(1.2345, locale='ar_EG', numbering_system='default')
u'1٫234'
>>> format_decimal(1.2345, locale='ar_EG', numbering_system='latn')
u'1.234'

The appropriate thousands grouping and the decimal separator are used for
each locale:

>>> format_decimal(12345.5, locale='en_US')
u'12,345.5'

By default the locale is allowed to truncate and round a high-precision
number by forcing its format pattern onto the decimal part. You can bypass
this behavior with the `decimal_quantization` parameter:

>>> format_decimal(1.2346, locale='en_US')
u'1.235'
>>> format_decimal(1.2346, locale='en_US', decimal_quantization=False)
u'1.2346'
>>> format_decimal(12345.67, locale='fr_CA', group_separator=False)
u'12345,67'
>>> format_decimal(12345.67, locale='en_US', group_separator=True)
u'12,345.67'

:param number: the number to format
:param format:
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param decimal_quantization: Truncate and round high-precision numbers to
                             the format pattern. Defaults to `True`.
:param group_separator: Boolean to switch group separator on/off in a locale's
                        number format.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: format_compact_decimal(number)

**Description:** Return the given decimal number formatted for a specific locale in compact form.

>>> format_compact_decimal(12345, format_type="short", locale='en_US')
u'12K'
>>> format_compact_decimal(12345, format_type="long", locale='en_US')
u'12 thousand'
>>> format_compact_decimal(12345, format_type="short", locale='en_US', fraction_digits=2)
u'12.34K'
>>> format_compact_decimal(1234567, format_type="short", locale="ja_JP")
u'123万'
>>> format_compact_decimal(2345678, format_type="long", locale="mk")
u'2 милиони'
>>> format_compact_decimal(21000000, format_type="long", locale="mk")
u'21 милион'
>>> format_compact_decimal(12345, format_type="short", locale='ar_EG', fraction_digits=2, numbering_system='default')
u'12٫34 ألف'

:param number: the number to format
:param format_type: Compact format to use ("short" or "long")
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param fraction_digits: Number of digits after the decimal point to use. Defaults to `0`.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: _get_compact_format(number, compact_format, locale, fraction_digits)

**Description:** Returns the number after dividing by the unit and the format pattern to use.
The algorithm is described here:
https://www.unicode.org/reports/tr35/tr35-45/tr35-numbers.html#Compact_Number_Formats.

## Class: UnknownCurrencyFormatError

**Description:** Exception raised when an unknown currency format is requested.

### Function: format_currency(number, currency, format, locale, currency_digits, format_type, decimal_quantization, group_separator)

**Description:** Return formatted currency value.

>>> format_currency(1099.98, 'USD', locale='en_US')
'$1,099.98'
>>> format_currency(1099.98, 'USD', locale='es_CO')
u'US$1.099,98'
>>> format_currency(1099.98, 'EUR', locale='de_DE')
u'1.099,98\xa0\u20ac'
>>> format_currency(1099.98, 'EGP', locale='ar_EG', numbering_system='default')
u'‏1٬099٫98 ج.م.‏'

The format can also be specified explicitly.  The currency is
placed with the '¤' sign.  As the sign gets repeated the format
expands (¤ being the symbol, ¤¤ is the currency abbreviation and
¤¤¤ is the full name of the currency):

>>> format_currency(1099.98, 'EUR', u'¤¤ #,##0.00', locale='en_US')
u'EUR 1,099.98'
>>> format_currency(1099.98, 'EUR', u'#,##0.00 ¤¤¤', locale='en_US')
u'1,099.98 euros'

Currencies usually have a specific number of decimal digits. This function
favours that information over the given format:

>>> format_currency(1099.98, 'JPY', locale='en_US')
u'\xa51,100'
>>> format_currency(1099.98, 'COP', u'#,##0.00', locale='es_ES')
u'1.099,98'

However, the number of decimal digits can be overridden from the currency
information, by setting the last parameter to ``False``:

>>> format_currency(1099.98, 'JPY', locale='en_US', currency_digits=False)
u'\xa51,099.98'
>>> format_currency(1099.98, 'COP', u'#,##0.00', locale='es_ES', currency_digits=False)
u'1.099,98'

If a format is not specified the type of currency format to use
from the locale can be specified:

>>> format_currency(1099.98, 'EUR', locale='en_US', format_type='standard')
u'\u20ac1,099.98'

When the given currency format type is not available, an exception is
raised:

>>> format_currency('1099.98', 'EUR', locale='root', format_type='unknown')
Traceback (most recent call last):
    ...
UnknownCurrencyFormatError: "'unknown' is not a known currency format type"

>>> format_currency(101299.98, 'USD', locale='en_US', group_separator=False)
u'$101299.98'

>>> format_currency(101299.98, 'USD', locale='en_US', group_separator=True)
u'$101,299.98'

You can also pass format_type='name' to use long display names. The order of
the number and currency name, along with the correct localized plural form
of the currency name, is chosen according to locale:

>>> format_currency(1, 'USD', locale='en_US', format_type='name')
u'1.00 US dollar'
>>> format_currency(1099.98, 'USD', locale='en_US', format_type='name')
u'1,099.98 US dollars'
>>> format_currency(1099.98, 'USD', locale='ee', format_type='name')
u'us ga dollar 1,099.98'

By default the locale is allowed to truncate and round a high-precision
number by forcing its format pattern onto the decimal part. You can bypass
this behavior with the `decimal_quantization` parameter:

>>> format_currency(1099.9876, 'USD', locale='en_US')
u'$1,099.99'
>>> format_currency(1099.9876, 'USD', locale='en_US', decimal_quantization=False)
u'$1,099.9876'

:param number: the number to format
:param currency: the currency code
:param format: the format string to use
:param locale: the `Locale` object or locale identifier.
               Defaults to the system currency locale or numeric locale.
:param currency_digits: use the currency's natural number of decimal digits
:param format_type: the currency format type to use
:param decimal_quantization: Truncate and round high-precision numbers to
                             the format pattern. Defaults to `True`.
:param group_separator: Boolean to switch group separator on/off in a locale's
                        number format.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: _format_currency_long_name(number, currency)

### Function: format_compact_currency(number, currency)

**Description:** Format a number as a currency value in compact form.

>>> format_compact_currency(12345, 'USD', locale='en_US')
u'$12K'
>>> format_compact_currency(123456789, 'USD', locale='en_US', fraction_digits=2)
u'$123.46M'
>>> format_compact_currency(123456789, 'EUR', locale='de_DE', fraction_digits=1)
'123,5 Mio. €'

:param number: the number to format
:param currency: the currency code
:param format_type: the compact format type to use. Defaults to "short".
:param locale: the `Locale` object or locale identifier.
               Defaults to the system currency locale or numeric locale.
:param fraction_digits: Number of digits after the decimal point to use. Defaults to `0`.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: format_percent(number, format, locale, decimal_quantization, group_separator)

**Description:** Return formatted percent value for a specific locale.

>>> format_percent(0.34, locale='en_US')
u'34%'
>>> format_percent(25.1234, locale='en_US')
u'2,512%'
>>> format_percent(25.1234, locale='sv_SE')
u'2\xa0512\xa0%'
>>> format_percent(25.1234, locale='ar_EG', numbering_system='default')
u'2٬512%'

The format pattern can also be specified explicitly:

>>> format_percent(25.1234, u'#,##0‰', locale='en_US')
u'25,123‰'

By default the locale is allowed to truncate and round a high-precision
number by forcing its format pattern onto the decimal part. You can bypass
this behavior with the `decimal_quantization` parameter:

>>> format_percent(23.9876, locale='en_US')
u'2,399%'
>>> format_percent(23.9876, locale='en_US', decimal_quantization=False)
u'2,398.76%'

>>> format_percent(229291.1234, locale='pt_BR', group_separator=False)
u'22929112%'

>>> format_percent(229291.1234, locale='pt_BR', group_separator=True)
u'22.929.112%'

:param number: the percent number to format
:param format:
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param decimal_quantization: Truncate and round high-precision numbers to
                             the format pattern. Defaults to `True`.
:param group_separator: Boolean to switch group separator on/off in a locale's
                        number format.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

### Function: format_scientific(number, format, locale, decimal_quantization)

**Description:** Return value formatted in scientific notation for a specific locale.

>>> format_scientific(10000, locale='en_US')
u'1E4'
>>> format_scientific(10000, locale='ar_EG', numbering_system='default')
u'1أس4'

The format pattern can also be specified explicitly:

>>> format_scientific(1234567, u'##0.##E00', locale='en_US')
u'1.23E06'

By default the locale is allowed to truncate and round a high-precision
number by forcing its format pattern onto the decimal part. You can bypass
this behavior with the `decimal_quantization` parameter:

>>> format_scientific(1234.9876, u'#.##E0', locale='en_US')
u'1.23E3'
>>> format_scientific(1234.9876, u'#.##E0', locale='en_US', decimal_quantization=False)
u'1.2349876E3'

:param number: the number to format
:param format:
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param decimal_quantization: Truncate and round high-precision numbers to
                             the format pattern. Defaults to `True`.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.

## Class: NumberFormatError

**Description:** Exception raised when a string cannot be parsed into a number.

### Function: parse_number(string, locale)

**Description:** Parse localized number string into an integer.

>>> parse_number('1,099', locale='en_US')
1099
>>> parse_number('1.099', locale='de_DE')
1099

When the given string cannot be parsed, an exception is raised:

>>> parse_number('1.099,98', locale='de')
Traceback (most recent call last):
    ...
NumberFormatError: '1.099,98' is not a valid number

:param string: the string to parse
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:return: the parsed number
:raise `NumberFormatError`: if the string can not be converted to a number
:raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.

### Function: parse_decimal(string, locale, strict)

**Description:** Parse localized decimal string into a decimal.

>>> parse_decimal('1,099.98', locale='en_US')
Decimal('1099.98')
>>> parse_decimal('1.099,98', locale='de')
Decimal('1099.98')
>>> parse_decimal('12 345,123', locale='ru')
Decimal('12345.123')
>>> parse_decimal('1٬099٫98', locale='ar_EG', numbering_system='default')
Decimal('1099.98')

When the given string cannot be parsed, an exception is raised:

>>> parse_decimal('2,109,998', locale='de')
Traceback (most recent call last):
    ...
NumberFormatError: '2,109,998' is not a valid decimal number

If `strict` is set to `True` and the given string contains a number
formatted in an irregular way, an exception is raised:

>>> parse_decimal('30.00', locale='de', strict=True)
Traceback (most recent call last):
    ...
NumberFormatError: '30.00' is not a properly formatted decimal number. Did you mean '3.000'? Or maybe '30,00'?

>>> parse_decimal('0.00', locale='de', strict=True)
Traceback (most recent call last):
    ...
NumberFormatError: '0.00' is not a properly formatted decimal number. Did you mean '0'?

:param string: the string to parse
:param locale: the `Locale` object or locale identifier. Defaults to the system numeric locale.
:param strict: controls whether numbers formatted in a weird way are
               accepted or rejected
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:raise NumberFormatError: if the string can not be converted to a
                          decimal number
:raise UnsupportedNumberingSystemError: if the numbering system is not supported by the locale.

### Function: _remove_trailing_zeros_after_decimal(string, decimal_symbol)

**Description:** Remove trailing zeros from the decimal part of a numeric string.

This function takes a string representing a numeric value and a decimal symbol.
It removes any trailing zeros that appear after the decimal symbol in the number.
If the decimal part becomes empty after removing trailing zeros, the decimal symbol
is also removed. If the string does not contain the decimal symbol, it is returned unchanged.

:param string: The numeric string from which to remove trailing zeros.
:type string: str
:param decimal_symbol: The symbol used to denote the decimal point.
:type decimal_symbol: str
:return: The numeric string with trailing zeros removed from its decimal part.
:rtype: str

Example:
>>> _remove_trailing_zeros_after_decimal("123.4500", ".")
'123.45'
>>> _remove_trailing_zeros_after_decimal("100.000", ".")
'100'
>>> _remove_trailing_zeros_after_decimal("100", ".")
'100'

### Function: parse_grouping(p)

**Description:** Parse primary and secondary digit grouping

>>> parse_grouping('##')
(1000, 1000)
>>> parse_grouping('#,###')
(3, 3)
>>> parse_grouping('#,####,###')
(3, 4)

### Function: parse_pattern(pattern)

**Description:** Parse number format patterns

## Class: NumberPattern

### Function: __init__(self, identifier)

**Description:** Create the exception.
:param identifier: the identifier string of the unsupported currency

### Function: _is_active(start, end)

### Function: __init__(self, message, suggestions)

### Function: _match_number(pattern)

### Function: parse_precision(p)

**Description:** Calculate the min and max allowed digits

### Function: __init__(self, pattern, prefix, suffix, grouping, int_prec, frac_prec, exp_prec, exp_plus, number_pattern)

### Function: __repr__(self)

### Function: compute_scale(self)

**Description:** Return the scaling factor to apply to the number before rendering.

Auto-set to a factor of 2 or 3 if presence of a ``%`` or ``‰`` sign is
detected in the prefix or suffix of the pattern. Default is to not mess
with the scale at all and keep it to 0.

### Function: scientific_notation_elements(self, value, locale)

**Description:** Returns normalized scientific notation components of a value.
        

### Function: apply(self, value, locale, currency, currency_digits, decimal_quantization, force_frac, group_separator)

**Description:** Renders into a string a number following the defined pattern.

Forced decimal quantization is active by default so we'll produce a
number string that is strictly following CLDR pattern definitions.

:param value: The value to format. If this is not a Decimal object,
              it will be cast to one.
:type value: decimal.Decimal|float|int
:param locale: The locale to use for formatting.
:type locale: str|babel.core.Locale
:param currency: Which currency, if any, to format as.
:type currency: str|None
:param currency_digits: Whether or not to use the currency's precision.
                        If false, the pattern's precision is used.
:type currency_digits: bool
:param decimal_quantization: Whether decimal numbers should be forcibly
                             quantized to produce a formatted output
                             strictly matching the CLDR definition for
                             the locale.
:type decimal_quantization: bool
:param force_frac: DEPRECATED - a forced override for `self.frac_prec`
                   for a single formatting invocation.
:param group_separator: Whether to use the locale's number group separator.
:param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                         The special value "default" will use the default numbering system of the locale.
:return: Formatted decimal string.
:rtype: str
:raise UnsupportedNumberingSystemError: If the numbering system is not supported by the locale.

### Function: _format_significant(self, value, minimum, maximum)

### Function: _format_int(self, value, min, max, locale)

### Function: _quantize_value(self, value, locale, frac_prec, group_separator)

### Function: _format_frac(self, value, locale, force_frac)
