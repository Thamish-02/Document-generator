## AI Summary

A file named isoparser.py.


### Function: _takes_ascii(f)

## Class: isoparser

### Function: func(self, str_in)

### Function: __init__(self, sep)

**Description:** :param sep:
    A single character that separates date and time portions. If
    ``None``, the parser will accept any single character.
    For strict ISO-8601 adherence, pass ``'T'``.

### Function: isoparse(self, dt_str)

**Description:** Parse an ISO-8601 datetime string into a :class:`datetime.datetime`.

An ISO-8601 datetime string consists of a date portion, followed
optionally by a time portion - the date and time portions are separated
by a single character separator, which is ``T`` in the official
standard. Incomplete date formats (such as ``YYYY-MM``) may *not* be
combined with a time portion.

Supported date formats are:

Common:

- ``YYYY``
- ``YYYY-MM``
- ``YYYY-MM-DD`` or ``YYYYMMDD``

Uncommon:

- ``YYYY-Www`` or ``YYYYWww`` - ISO week (day defaults to 0)
- ``YYYY-Www-D`` or ``YYYYWwwD`` - ISO week and day

The ISO week and day numbering follows the same logic as
:func:`datetime.date.isocalendar`.

Supported time formats are:

- ``hh``
- ``hh:mm`` or ``hhmm``
- ``hh:mm:ss`` or ``hhmmss``
- ``hh:mm:ss.ssssss`` (Up to 6 sub-second digits)

Midnight is a special case for `hh`, as the standard supports both
00:00 and 24:00 as a representation. The decimal separator can be
either a dot or a comma.


.. caution::

    Support for fractional components other than seconds is part of the
    ISO-8601 standard, but is not currently implemented in this parser.

Supported time zone offset formats are:

- `Z` (UTC)
- `±HH:MM`
- `±HHMM`
- `±HH`

Offsets will be represented as :class:`dateutil.tz.tzoffset` objects,
with the exception of UTC, which will be represented as
:class:`dateutil.tz.tzutc`. Time zone offsets equivalent to UTC (such
as `+00:00`) will also be represented as :class:`dateutil.tz.tzutc`.

:param dt_str:
    A string or stream containing only an ISO-8601 datetime string

:return:
    Returns a :class:`datetime.datetime` representing the string.
    Unspecified components default to their lowest value.

.. warning::

    As of version 2.7.0, the strictness of the parser should not be
    considered a stable part of the contract. Any valid ISO-8601 string
    that parses correctly with the default settings will continue to
    parse correctly in future versions, but invalid strings that
    currently fail (e.g. ``2017-01-01T00:00+00:00:00``) are not
    guaranteed to continue failing in future versions if they encode
    a valid date.

.. versionadded:: 2.7.0

### Function: parse_isodate(self, datestr)

**Description:** Parse the date portion of an ISO string.

:param datestr:
    The string portion of an ISO string, without a separator

:return:
    Returns a :class:`datetime.date` object

### Function: parse_isotime(self, timestr)

**Description:** Parse the time portion of an ISO string.

:param timestr:
    The time portion of an ISO string, without a separator

:return:
    Returns a :class:`datetime.time` object

### Function: parse_tzstr(self, tzstr, zero_as_utc)

**Description:** Parse a valid ISO time zone string.

See :func:`isoparser.isoparse` for details on supported formats.

:param tzstr:
    A string representing an ISO time zone offset

:param zero_as_utc:
    Whether to return :class:`dateutil.tz.tzutc` for zero-offset zones

:return:
    Returns :class:`dateutil.tz.tzoffset` for offsets and
    :class:`dateutil.tz.tzutc` for ``Z`` and (if ``zero_as_utc`` is
    specified) offsets equivalent to UTC.

### Function: _parse_isodate(self, dt_str)

### Function: _parse_isodate_common(self, dt_str)

### Function: _parse_isodate_uncommon(self, dt_str)

### Function: _calculate_weekdate(self, year, week, day)

**Description:** Calculate the day of corresponding to the ISO year-week-day calendar.

This function is effectively the inverse of
:func:`datetime.date.isocalendar`.

:param year:
    The year in the ISO calendar

:param week:
    The week in the ISO calendar - range is [1, 53]

:param day:
    The day in the ISO calendar - range is [1 (MON), 7 (SUN)]

:return:
    Returns a :class:`datetime.date`

### Function: _parse_isotime(self, timestr)

### Function: _parse_tzstr(self, tzstr, zero_as_utc)
