## AI Summary

A file named _common.py.


### Function: tzname_in_python2(namefunc)

**Description:** Change unicode output into bytestrings in Python 2

tzname() API changed in Python 3. It used to return bytes, but was changed
to unicode strings

### Function: _validate_fromutc_inputs(f)

**Description:** The CPython version of ``fromutc`` checks that the input is a ``datetime``
object and that ``self`` is attached as its ``tzinfo``.

## Class: _tzinfo

**Description:** Base class for all ``dateutil`` ``tzinfo`` objects.

## Class: tzrangebase

**Description:** This is an abstract base class for time zones represented by an annual
transition into and out of DST. Child classes should implement the following
methods:

    * ``__init__(self, *args, **kwargs)``
    * ``transitions(self, year)`` - this is expected to return a tuple of
      datetimes representing the DST on and off transitions in standard
      time.

A fully initialized ``tzrangebase`` subclass should also provide the
following attributes:
    * ``hasdst``: Boolean whether or not the zone uses DST.
    * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects
      representing the respective UTC offsets.
    * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short
      abbreviations in DST and STD, respectively.
    * ``_hasdst``: Whether or not the zone has DST.

.. versionadded:: 2.6.0

### Function: enfold(dt, fold)

**Description:** Provides a unified interface for assigning the ``fold`` attribute to
datetimes both before and after the implementation of PEP-495.

:param fold:
    The value for the ``fold`` attribute in the returned datetime. This
    should be either 0 or 1.

:return:
    Returns an object for which ``getattr(dt, 'fold', 0)`` returns
    ``fold`` for all versions of Python. In versions prior to
    Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
    subclass of :py:class:`datetime.datetime` with the ``fold``
    attribute added, if ``fold`` is 1.

.. versionadded:: 2.6.0

## Class: _DatetimeWithFold

**Description:** This is a class designed to provide a PEP 495-compliant interface for
Python versions before 3.6. It is used only for dates in a fold, so
the ``fold`` attribute is fixed at ``1``.

.. versionadded:: 2.6.0

### Function: enfold(dt, fold)

**Description:** Provides a unified interface for assigning the ``fold`` attribute to
datetimes both before and after the implementation of PEP-495.

:param fold:
    The value for the ``fold`` attribute in the returned datetime. This
    should be either 0 or 1.

:return:
    Returns an object for which ``getattr(dt, 'fold', 0)`` returns
    ``fold`` for all versions of Python. In versions prior to
    Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
    subclass of :py:class:`datetime.datetime` with the ``fold``
    attribute added, if ``fold`` is 1.

.. versionadded:: 2.6.0

### Function: fromutc(self, dt)

### Function: is_ambiguous(self, dt)

**Description:** Whether or not the "wall time" of a given datetime is ambiguous in this
zone.

:param dt:
    A :py:class:`datetime.datetime`, naive or time zone aware.


:return:
    Returns ``True`` if ambiguous, ``False`` otherwise.

.. versionadded:: 2.6.0

### Function: _fold_status(self, dt_utc, dt_wall)

**Description:** Determine the fold status of a "wall" datetime, given a representation
of the same datetime as a (naive) UTC datetime. This is calculated based
on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
datetimes, and that this offset is the actual number of hours separating
``dt_utc`` and ``dt_wall``.

:param dt_utc:
    Representation of the datetime as UTC

:param dt_wall:
    Representation of the datetime as "wall time". This parameter must
    either have a `fold` attribute or have a fold-naive
    :class:`datetime.tzinfo` attached, otherwise the calculation may
    fail.

### Function: _fold(self, dt)

### Function: _fromutc(self, dt)

**Description:** Given a timezone-aware datetime in a given timezone, calculates a
timezone-aware datetime in a new timezone.

Since this is the one time that we *know* we have an unambiguous
datetime object, we take this opportunity to determine whether the
datetime is ambiguous and in a "fold" state (e.g. if it's the first
occurrence, chronologically, of the ambiguous datetime).

:param dt:
    A timezone-aware :class:`datetime.datetime` object.

### Function: fromutc(self, dt)

**Description:** Given a timezone-aware datetime in a given timezone, calculates a
timezone-aware datetime in a new timezone.

Since this is the one time that we *know* we have an unambiguous
datetime object, we take this opportunity to determine whether the
datetime is ambiguous and in a "fold" state (e.g. if it's the first
occurrence, chronologically, of the ambiguous datetime).

:param dt:
    A timezone-aware :class:`datetime.datetime` object.

### Function: __init__(self)

### Function: utcoffset(self, dt)

### Function: dst(self, dt)

### Function: tzname(self, dt)

### Function: fromutc(self, dt)

**Description:** Given a datetime in UTC, return local time 

### Function: is_ambiguous(self, dt)

**Description:** Whether or not the "wall time" of a given datetime is ambiguous in this
zone.

:param dt:
    A :py:class:`datetime.datetime`, naive or time zone aware.


:return:
    Returns ``True`` if ambiguous, ``False`` otherwise.

.. versionadded:: 2.6.0

### Function: _isdst(self, dt)

### Function: _naive_isdst(self, dt, transitions)

### Function: _dst_base_offset(self)

### Function: __ne__(self, other)

### Function: __repr__(self)

### Function: adjust_encoding()

### Function: replace(self)

**Description:** Return a datetime with the same attributes, except for those
attributes given new values by whichever keyword arguments are
specified. Note that tzinfo=None can be specified to create a naive
datetime from an aware datetime with no conversion of date and time
data.

This is reimplemented in ``_DatetimeWithFold`` because pypy3 will
return a ``datetime.datetime`` even if ``fold`` is unchanged.

### Function: fold(self)
