## AI Summary

A file named rrule.py.


## Class: weekday

**Description:** This version of weekday does not allow n = 0.

### Function: _invalidates_cache(f)

**Description:** Decorator for rruleset methods which may invalidate the
cached length.

## Class: rrulebase

## Class: rrule

**Description:** That's the base of the rrule operation. It accepts all the keywords
defined in the RFC as its constructor parameters (except byday,
which was renamed to byweekday) and more. The constructor prototype is::

        rrule(freq)

Where freq must be one of YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
or SECONDLY.

.. note::
    Per RFC section 3.3.10, recurrence instances falling on invalid dates
    and times are ignored rather than coerced:

        Recurrence rules may generate recurrence instances with an invalid
        date (e.g., February 30) or nonexistent local time (e.g., 1:30 AM
        on a day where the local time is moved forward by an hour at 1:00
        AM).  Such recurrence instances MUST be ignored and MUST NOT be
        counted as part of the recurrence set.

    This can lead to possibly surprising behavior when, for example, the
    start date occurs at the end of the month:

    >>> from dateutil.rrule import rrule, MONTHLY
    >>> from datetime import datetime
    >>> start_date = datetime(2014, 12, 31)
    >>> list(rrule(freq=MONTHLY, count=4, dtstart=start_date))
    ... # doctest: +NORMALIZE_WHITESPACE
    [datetime.datetime(2014, 12, 31, 0, 0),
     datetime.datetime(2015, 1, 31, 0, 0),
     datetime.datetime(2015, 3, 31, 0, 0),
     datetime.datetime(2015, 5, 31, 0, 0)]

Additionally, it supports the following keyword arguments:

:param dtstart:
    The recurrence start. Besides being the base for the recurrence,
    missing parameters in the final recurrence instances will also be
    extracted from this date. If not given, datetime.now() will be used
    instead.
:param interval:
    The interval between each freq iteration. For example, when using
    YEARLY, an interval of 2 means once every two years, but with HOURLY,
    it means once every two hours. The default interval is 1.
:param wkst:
    The week start day. Must be one of the MO, TU, WE constants, or an
    integer, specifying the first day of the week. This will affect
    recurrences based on weekly periods. The default week start is got
    from calendar.firstweekday(), and may be modified by
    calendar.setfirstweekday().
:param count:
    If given, this determines how many occurrences will be generated.

    .. note::
        As of version 2.5.0, the use of the keyword ``until`` in conjunction
        with ``count`` is deprecated, to make sure ``dateutil`` is fully
        compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
        html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
        **must not** occur in the same call to ``rrule``.
:param until:
    If given, this must be a datetime instance specifying the upper-bound
    limit of the recurrence. The last recurrence in the rule is the greatest
    datetime that is less than or equal to the value specified in the
    ``until`` parameter.

    .. note::
        As of version 2.5.0, the use of the keyword ``until`` in conjunction
        with ``count`` is deprecated, to make sure ``dateutil`` is fully
        compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
        html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
        **must not** occur in the same call to ``rrule``.
:param bysetpos:
    If given, it must be either an integer, or a sequence of integers,
    positive or negative. Each given integer will specify an occurrence
    number, corresponding to the nth occurrence of the rule inside the
    frequency period. For example, a bysetpos of -1 if combined with a
    MONTHLY frequency, and a byweekday of (MO, TU, WE, TH, FR), will
    result in the last work day of every month.
:param bymonth:
    If given, it must be either an integer, or a sequence of integers,
    meaning the months to apply the recurrence to.
:param bymonthday:
    If given, it must be either an integer, or a sequence of integers,
    meaning the month days to apply the recurrence to.
:param byyearday:
    If given, it must be either an integer, or a sequence of integers,
    meaning the year days to apply the recurrence to.
:param byeaster:
    If given, it must be either an integer, or a sequence of integers,
    positive or negative. Each integer will define an offset from the
    Easter Sunday. Passing the offset 0 to byeaster will yield the Easter
    Sunday itself. This is an extension to the RFC specification.
:param byweekno:
    If given, it must be either an integer, or a sequence of integers,
    meaning the week numbers to apply the recurrence to. Week numbers
    have the meaning described in ISO8601, that is, the first week of
    the year is that containing at least four days of the new year.
:param byweekday:
    If given, it must be either an integer (0 == MO), a sequence of
    integers, one of the weekday constants (MO, TU, etc), or a sequence
    of these constants. When given, these variables will define the
    weekdays where the recurrence will be applied. It's also possible to
    use an argument n for the weekday instances, which will mean the nth
    occurrence of this weekday in the period. For example, with MONTHLY,
    or with YEARLY and BYMONTH, using FR(+1) in byweekday will specify the
    first friday of the month where the recurrence happens. Notice that in
    the RFC documentation, this is specified as BYDAY, but was renamed to
    avoid the ambiguity of that keyword.
:param byhour:
    If given, it must be either an integer, or a sequence of integers,
    meaning the hours to apply the recurrence to.
:param byminute:
    If given, it must be either an integer, or a sequence of integers,
    meaning the minutes to apply the recurrence to.
:param bysecond:
    If given, it must be either an integer, or a sequence of integers,
    meaning the seconds to apply the recurrence to.
:param cache:
    If given, it must be a boolean value specifying to enable or disable
    caching of results. If you will use the same rrule instance multiple
    times, enabling caching will improve the performance considerably.
 

## Class: _iterinfo

## Class: rruleset

**Description:** The rruleset type allows more complex recurrence setups, mixing
multiple rules, dates, exclusion rules, and exclusion dates. The type
constructor takes the following keyword arguments:

:param cache: If True, caching of results will be enabled, improving
              performance of multiple queries considerably. 

## Class: _rrulestr

**Description:** Parses a string representation of a recurrence rule or set of
recurrence rules.

:param s:
    Required, a string defining one or more recurrence rules.

:param dtstart:
    If given, used as the default recurrence start if not specified in the
    rule string.

:param cache:
    If set ``True`` caching of results will be enabled, improving
    performance of multiple queries considerably.

:param unfold:
    If set ``True`` indicates that a rule string is split over more
    than one line and should be joined before processing.

:param forceset:
    If set ``True`` forces a :class:`dateutil.rrule.rruleset` to
    be returned.

:param compatible:
    If set ``True`` forces ``unfold`` and ``forceset`` to be ``True``.

:param ignoretz:
    If set ``True``, time zones in parsed strings are ignored and a naive
    :class:`datetime.datetime` object is returned.

:param tzids:
    If given, a callable or mapping used to retrieve a
    :class:`datetime.tzinfo` from a string representation.
    Defaults to :func:`dateutil.tz.gettz`.

:param tzinfos:
    Additional time zone names / aliases which may be present in a string
    representation.  See :func:`dateutil.parser.parse` for more
    information.

:return:
    Returns a :class:`dateutil.rrule.rruleset` or
    :class:`dateutil.rrule.rrule`

### Function: __init__(self, wkday, n)

### Function: inner_func(self)

### Function: __init__(self, cache)

### Function: __iter__(self)

### Function: _invalidate_cache(self)

### Function: _iter_cached(self)

### Function: __getitem__(self, item)

### Function: __contains__(self, item)

### Function: count(self)

**Description:** Returns the number of recurrences in this set. It will have go
through the whole recurrence, if this hasn't been done before. 

### Function: before(self, dt, inc)

**Description:** Returns the last recurrence before the given datetime instance. The
inc keyword defines what happens if dt is an occurrence. With
inc=True, if dt itself is an occurrence, it will be returned. 

### Function: after(self, dt, inc)

**Description:** Returns the first recurrence after the given datetime instance. The
inc keyword defines what happens if dt is an occurrence. With
inc=True, if dt itself is an occurrence, it will be returned.  

### Function: xafter(self, dt, count, inc)

**Description:** Generator which yields up to `count` recurrences after the given
datetime instance, equivalent to `after`.

:param dt:
    The datetime at which to start generating recurrences.

:param count:
    The maximum number of recurrences to generate. If `None` (default),
    dates are generated until the recurrence rule is exhausted.

:param inc:
    If `dt` is an instance of the rule and `inc` is `True`, it is
    included in the output.

:yields: Yields a sequence of `datetime` objects.

### Function: between(self, after, before, inc, count)

**Description:** Returns all the occurrences of the rrule between after and before.
The inc keyword defines what happens if after and/or before are
themselves occurrences. With inc=True, they will be included in the
list, if they are found in the recurrence set. 

### Function: __init__(self, freq, dtstart, interval, wkst, count, until, bysetpos, bymonth, bymonthday, byyearday, byeaster, byweekno, byweekday, byhour, byminute, bysecond, cache)

### Function: __str__(self)

**Description:** Output a string that would generate this RRULE if passed to rrulestr.
This is mostly compatible with RFC5545, except for the
dateutil-specific extension BYEASTER.

### Function: replace(self)

**Description:** Return new rrule with same attributes except for those attributes given new
values by whichever keyword arguments are specified.

### Function: _iter(self)

### Function: __construct_byset(self, start, byxxx, base)

**Description:** If a `BYXXX` sequence is passed to the constructor at the same level as
`FREQ` (e.g. `FREQ=HOURLY,BYHOUR={2,4,7},INTERVAL=3`), there are some
specifications which cannot be reached given some starting conditions.

This occurs whenever the interval is not coprime with the base of a
given unit and the difference between the starting position and the
ending position is not coprime with the greatest common denominator
between the interval and the base. For example, with a FREQ of hourly
starting at 17:00 and an interval of 4, the only valid values for
BYHOUR would be {21, 1, 5, 9, 13, 17}, because 4 and 24 are not
coprime.

:param start:
    Specifies the starting position.
:param byxxx:
    An iterable containing the list of allowed values.
:param base:
    The largest allowable value for the specified frequency (e.g.
    24 hours, 60 minutes).

This does not preserve the type of the iterable, returning a set, since
the values should be unique and the order is irrelevant, this will
speed up later lookups.

In the event of an empty set, raises a :exception:`ValueError`, as this
results in an empty rrule.

### Function: __mod_distance(self, value, byxxx, base)

**Description:** Calculates the next value in a sequence where the `FREQ` parameter is
specified along with a `BYXXX` parameter at the same "level"
(e.g. `HOURLY` specified with `BYHOUR`).

:param value:
    The old value of the component.
:param byxxx:
    The `BYXXX` set, which should have been generated by
    `rrule._construct_byset`, or something else which checks that a
    valid rule is present.
:param base:
    The largest allowable value for the specified frequency (e.g.
    24 hours, 60 minutes).

If a valid value is not found after `base` iterations (the maximum
number before the sequence would start to repeat), this raises a
:exception:`ValueError`, as no valid values were found.

This returns a tuple of `divmod(n*interval, base)`, where `n` is the
smallest number of `interval` repetitions until the next specified
value in `byxxx` is found.

### Function: __init__(self, rrule)

### Function: rebuild(self, year, month)

### Function: ydayset(self, year, month, day)

### Function: mdayset(self, year, month, day)

### Function: wdayset(self, year, month, day)

### Function: ddayset(self, year, month, day)

### Function: htimeset(self, hour, minute, second)

### Function: mtimeset(self, hour, minute, second)

### Function: stimeset(self, hour, minute, second)

## Class: _genitem

### Function: __init__(self, cache)

### Function: rrule(self, rrule)

**Description:** Include the given :py:class:`rrule` instance in the recurrence set
generation. 

### Function: rdate(self, rdate)

**Description:** Include the given :py:class:`datetime` instance in the recurrence
set generation. 

### Function: exrule(self, exrule)

**Description:** Include the given rrule instance in the recurrence set exclusion
list. Dates which are part of the given recurrence rules will not
be generated, even if some inclusive rrule or rdate matches them.

### Function: exdate(self, exdate)

**Description:** Include the given datetime instance in the recurrence set
exclusion list. Dates included that way will not be generated,
even if some inclusive rrule or rdate matches them. 

### Function: _iter(self)

### Function: _handle_int(self, rrkwargs, name, value)

### Function: _handle_int_list(self, rrkwargs, name, value)

### Function: _handle_FREQ(self, rrkwargs, name, value)

### Function: _handle_UNTIL(self, rrkwargs, name, value)

### Function: _handle_WKST(self, rrkwargs, name, value)

### Function: _handle_BYWEEKDAY(self, rrkwargs, name, value)

**Description:** Two ways to specify this: +1MO or MO(+1)

### Function: _parse_rfc_rrule(self, line, dtstart, cache, ignoretz, tzinfos)

### Function: _parse_date_value(self, date_value, parms, rule_tzids, ignoretz, tzids, tzinfos)

### Function: _parse_rfc(self, s, dtstart, cache, unfold, forceset, compatible, ignoretz, tzids, tzinfos)

### Function: __call__(self, s)

### Function: __init__(self, genlist, gen)

### Function: __next__(self)

### Function: __lt__(self, other)

### Function: __gt__(self, other)

### Function: __eq__(self, other)

### Function: __ne__(self, other)
