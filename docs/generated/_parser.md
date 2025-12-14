## AI Summary

A file named _parser.py.


## Class: _timelex

## Class: _resultbase

## Class: parserinfo

**Description:** Class which handles what inputs are accepted. Subclass this to customize
the language and acceptable values for each parameter.

:param dayfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the day (``True``) or month (``False``). If
    ``yearfirst`` is set to ``True``, this distinguishes between YDM
    and YMD. Default is ``False``.

:param yearfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the year. If ``True``, the first number is taken
    to be the year, otherwise the last number is taken to be the year.
    Default is ``False``.

## Class: _ymd

## Class: parser

### Function: parse(timestr, parserinfo)

**Description:** Parse a string in one of the supported formats, using the
``parserinfo`` parameters.

:param timestr:
    A string containing a date/time stamp.

:param parserinfo:
    A :class:`parserinfo` object containing parameters for the parser.
    If ``None``, the default arguments to the :class:`parserinfo`
    constructor are used.

The ``**kwargs`` parameter takes the following keyword arguments:

:param default:
    The default datetime object, if this is a datetime object and not
    ``None``, elements specified in ``timestr`` replace elements in the
    default object.

:param ignoretz:
    If set ``True``, time zones in parsed strings are ignored and a naive
    :class:`datetime` object is returned.

:param tzinfos:
    Additional time zone names / aliases which may be present in the
    string. This argument maps time zone names (and optionally offsets
    from those time zones) to time zones. This parameter can be a
    dictionary with timezone aliases mapping time zone names to time
    zones or a function taking two parameters (``tzname`` and
    ``tzoffset``) and returning a time zone.

    The timezones to which the names are mapped can be an integer
    offset from UTC in seconds or a :class:`tzinfo` object.

    .. doctest::
       :options: +NORMALIZE_WHITESPACE

        >>> from dateutil.parser import parse
        >>> from dateutil.tz import gettz
        >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
        >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
        >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21,
                          tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

    This parameter is ignored if ``ignoretz`` is set.

:param dayfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the day (``True``) or month (``False``). If
    ``yearfirst`` is set to ``True``, this distinguishes between YDM and
    YMD. If set to ``None``, this value is retrieved from the current
    :class:`parserinfo` object (which itself defaults to ``False``).

:param yearfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
    be the year, otherwise the last number is taken to be the year. If
    this is set to ``None``, the value is retrieved from the current
    :class:`parserinfo` object (which itself defaults to ``False``).

:param fuzzy:
    Whether to allow fuzzy parsing, allowing for string like "Today is
    January 1, 2047 at 8:21:00AM".

:param fuzzy_with_tokens:
    If ``True``, ``fuzzy`` is automatically set to True, and the parser
    will return a tuple where the first element is the parsed
    :class:`datetime.datetime` datetimestamp and the second element is
    a tuple containing the portions of the string which were ignored:

    .. doctest::

        >>> from dateutil.parser import parse
        >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
        (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

:return:
    Returns a :class:`datetime.datetime` object or, if the
    ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
    first element being a :class:`datetime.datetime` object, the second
    a tuple containing the fuzzy tokens.

:raises ParserError:
    Raised for invalid or unknown string formats, if the provided
    :class:`tzinfo` is not in a valid format, or if an invalid date would
    be created.

:raises OverflowError:
    Raised if the parsed date exceeds the largest valid C integer on
    your system.

## Class: _tzparser

### Function: _parsetz(tzstr)

## Class: ParserError

**Description:** Exception subclass used for any failure to parse a datetime string.

This is a subclass of :py:exc:`ValueError`, and should be raised any time
earlier versions of ``dateutil`` would have raised ``ValueError``.

.. versionadded:: 2.8.1

## Class: UnknownTimezoneWarning

**Description:** Raised when the parser finds a timezone it cannot parse into a tzinfo.

.. versionadded:: 2.7.0

### Function: __init__(self, instream)

### Function: get_token(self)

**Description:** This function breaks the time string into lexical units (tokens), which
can be parsed by the parser. Lexical units are demarcated by changes in
the character set, so any continuous string of letters is considered
one unit, any continuous string of numbers is considered one unit.

The main complication arises from the fact that dots ('.') can be used
both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
"4:30:21.447"). As such, it is necessary to read the full context of
any dot-separated strings before breaking it into tokens; as such, this
function maintains a "token stack", for when the ambiguous context
demands that multiple tokens be parsed at once.

### Function: __iter__(self)

### Function: __next__(self)

### Function: next(self)

### Function: split(cls, s)

### Function: isword(cls, nextchar)

**Description:** Whether or not the next character is part of a word 

### Function: isnum(cls, nextchar)

**Description:** Whether the next character is part of a number 

### Function: isspace(cls, nextchar)

**Description:** Whether the next character is whitespace 

### Function: __init__(self)

### Function: _repr(self, classname)

### Function: __len__(self)

### Function: __repr__(self)

### Function: __init__(self, dayfirst, yearfirst)

### Function: _convert(self, lst)

### Function: jump(self, name)

### Function: weekday(self, name)

### Function: month(self, name)

### Function: hms(self, name)

### Function: ampm(self, name)

### Function: pertain(self, name)

### Function: utczone(self, name)

### Function: tzoffset(self, name)

### Function: convertyear(self, year, century_specified)

**Description:** Converts two-digit years to year within [-50, 49]
range of self._year (current local time)

### Function: validate(self, res)

### Function: __init__(self)

### Function: has_year(self)

### Function: has_month(self)

### Function: has_day(self)

### Function: could_be_day(self, value)

### Function: append(self, val, label)

### Function: _resolve_from_stridxs(self, strids)

**Description:** Try to resolve the identities of year/month/day elements using
ystridx, mstridx, and dstridx, if enough of these are specified.

### Function: resolve_ymd(self, yearfirst, dayfirst)

### Function: __init__(self, info)

### Function: parse(self, timestr, default, ignoretz, tzinfos)

**Description:** Parse the date/time string into a :class:`datetime.datetime` object.

:param timestr:
    Any date/time string using the supported formats.

:param default:
    The default datetime object, if this is a datetime object and not
    ``None``, elements specified in ``timestr`` replace elements in the
    default object.

:param ignoretz:
    If set ``True``, time zones in parsed strings are ignored and a
    naive :class:`datetime.datetime` object is returned.

:param tzinfos:
    Additional time zone names / aliases which may be present in the
    string. This argument maps time zone names (and optionally offsets
    from those time zones) to time zones. This parameter can be a
    dictionary with timezone aliases mapping time zone names to time
    zones or a function taking two parameters (``tzname`` and
    ``tzoffset``) and returning a time zone.

    The timezones to which the names are mapped can be an integer
    offset from UTC in seconds or a :class:`tzinfo` object.

    .. doctest::
       :options: +NORMALIZE_WHITESPACE

        >>> from dateutil.parser import parse
        >>> from dateutil.tz import gettz
        >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
        >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
        >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21,
                          tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

    This parameter is ignored if ``ignoretz`` is set.

:param \*\*kwargs:
    Keyword arguments as passed to ``_parse()``.

:return:
    Returns a :class:`datetime.datetime` object or, if the
    ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
    first element being a :class:`datetime.datetime` object, the second
    a tuple containing the fuzzy tokens.

:raises ParserError:
    Raised for invalid or unknown string format, if the provided
    :class:`tzinfo` is not in a valid format, or if an invalid date
    would be created.

:raises TypeError:
    Raised for non-string or character stream input.

:raises OverflowError:
    Raised if the parsed date exceeds the largest valid C integer on
    your system.

## Class: _result

### Function: _parse(self, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens)

**Description:** Private method which performs the heavy lifting of parsing, called from
``parse()``, which passes on its ``kwargs`` to this function.

:param timestr:
    The string to parse.

:param dayfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the day (``True``) or month (``False``). If
    ``yearfirst`` is set to ``True``, this distinguishes between YDM
    and YMD. If set to ``None``, this value is retrieved from the
    current :class:`parserinfo` object (which itself defaults to
    ``False``).

:param yearfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the year. If ``True``, the first number is taken
    to be the year, otherwise the last number is taken to be the year.
    If this is set to ``None``, the value is retrieved from the current
    :class:`parserinfo` object (which itself defaults to ``False``).

:param fuzzy:
    Whether to allow fuzzy parsing, allowing for string like "Today is
    January 1, 2047 at 8:21:00AM".

:param fuzzy_with_tokens:
    If ``True``, ``fuzzy`` is automatically set to True, and the parser
    will return a tuple where the first element is the parsed
    :class:`datetime.datetime` datetimestamp and the second element is
    a tuple containing the portions of the string which were ignored:

    .. doctest::

        >>> from dateutil.parser import parse
        >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
        (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

### Function: _parse_numeric_token(self, tokens, idx, info, ymd, res, fuzzy)

### Function: _find_hms_idx(self, idx, tokens, info, allow_jump)

### Function: _assign_hms(self, res, value_repr, hms)

### Function: _could_be_tzname(self, hour, tzname, tzoffset, token)

### Function: _ampm_valid(self, hour, ampm, fuzzy)

**Description:** For fuzzy parsing, 'a' or 'am' (both valid English words)
may erroneously trigger the AM/PM flag. Deal with that
here.

### Function: _adjust_ampm(self, hour, ampm)

### Function: _parse_min_sec(self, value)

### Function: _parse_hms(self, idx, tokens, info, hms_idx)

### Function: _parsems(self, value)

**Description:** Parse a I[.F] seconds value into (seconds, microseconds).

### Function: _to_decimal(self, val)

### Function: _build_tzinfo(self, tzinfos, tzname, tzoffset)

### Function: _build_tzaware(self, naive, res, tzinfos)

### Function: _build_naive(self, res, default)

### Function: _assign_tzname(self, dt, tzname)

### Function: _recombine_skipped(self, tokens, skipped_idxs)

**Description:** >>> tokens = ["foo", " ", "bar", " ", "19June2000", "baz"]
>>> skipped_idxs = [0, 1, 2, 5]
>>> _recombine_skipped(tokens, skipped_idxs)
["foo bar", "baz"]

## Class: _result

### Function: parse(self, tzstr)

### Function: __str__(self)

### Function: __repr__(self)

## Class: _attr

### Function: __repr__(self)

### Function: __init__(self)
