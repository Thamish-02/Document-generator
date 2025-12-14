## AI Summary

A file named _locales.py.


### Function: find_comma_decimal_point_locale()

**Description:** See if platform has a decimal point as comma locale.

Find a locale that uses a comma instead of a period as the
decimal point.

Returns
-------
old_locale: str
    Locale when the function was called.
new_locale: {str, None)
    First French locale found, None if none found.

## Class: CommaDecimalPointLocale

**Description:** Sets LC_NUMERIC to a locale with comma as decimal point.

Classes derived from this class have setup and teardown methods that run
tests with locale.LC_NUMERIC set to a locale where commas (',') are used as
the decimal point instead of periods ('.'). On exit the locale is restored
to the initial locale. It also serves as context manager with the same
effect. If no such locale is available, the test is skipped.

### Function: setup_method(self)

### Function: teardown_method(self)

### Function: __enter__(self)

### Function: __exit__(self, type, value, traceback)
