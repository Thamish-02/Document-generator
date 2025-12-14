## AI Summary

A file named win.py.


### Function: _settzkeyname()

## Class: tzres

**Description:** Class for accessing ``tzres.dll``, which contains timezone name related
resources.

.. versionadded:: 2.5.0

## Class: tzwinbase

**Description:** tzinfo class based on win32's timezones available in the registry.

## Class: tzwin

**Description:** Time zone object created from the zone info in the Windows registry

These are similar to :py:class:`dateutil.tz.tzrange` objects in that
the time zone data is provided in the format of a single offset rule
for either 0 or 2 time zone transitions per year.

:param: name
    The name of a Windows time zone key, e.g. "Eastern Standard Time".
    The full list of keys can be retrieved with :func:`tzwin.list`.

## Class: tzwinlocal

**Description:** Class representing the local time zone information in the Windows registry

While :class:`dateutil.tz.tzlocal` makes system calls (via the :mod:`time`
module) to retrieve time zone information, ``tzwinlocal`` retrieves the
rules directly from the Windows registry and creates an object like
:class:`dateutil.tz.tzwin`.

Because Windows does not have an equivalent of :func:`time.tzset`, on
Windows, :class:`dateutil.tz.tzlocal` instances will always reflect the
time zone settings *at the time that the process was started*, meaning
changes to the machine's time zone settings during the run of a program
on Windows will **not** be reflected by :class:`dateutil.tz.tzlocal`.
Because ``tzwinlocal`` reads the registry directly, it is unaffected by
this issue.

### Function: picknthweekday(year, month, dayofweek, hour, minute, whichweek)

**Description:** dayofweek == 0 means Sunday, whichweek 5 means last instance 

### Function: valuestodict(key)

**Description:** Convert a registry key's values to a dictionary.

### Function: __init__(self, tzres_loc)

### Function: load_name(self, offset)

**Description:** Load a timezone name from a DLL offset (integer).

>>> from dateutil.tzwin import tzres
>>> tzr = tzres()
>>> print(tzr.load_name(112))
'Eastern Standard Time'

:param offset:
    A positive integer value referring to a string from the tzres dll.

.. note::

    Offsets found in the registry are generally of the form
    ``@tzres.dll,-114``. The offset in this case is 114, not -114.

### Function: name_from_string(self, tzname_str)

**Description:** Parse strings as returned from the Windows registry into the time zone
name as defined in the registry.

>>> from dateutil.tzwin import tzres
>>> tzr = tzres()
>>> print(tzr.name_from_string('@tzres.dll,-251'))
'Dateline Daylight Time'
>>> print(tzr.name_from_string('Eastern Standard Time'))
'Eastern Standard Time'

:param tzname_str:
    A timezone name string as returned from a Windows registry key.

:return:
    Returns the localized timezone string from tzres.dll if the string
    is of the form `@tzres.dll,-offset`, else returns the input string.

### Function: __init__(self)

### Function: __eq__(self, other)

### Function: list()

**Description:** Return a list of all time zones known to the system.

### Function: display(self)

**Description:** Return the display name of the time zone.

### Function: transitions(self, year)

**Description:** For a given year, get the DST on and off transition times, expressed
always on the standard time side. For zones with no transitions, this
function returns ``None``.

:param year:
    The year whose transitions you would like to query.

:return:
    Returns a :class:`tuple` of :class:`datetime.datetime` objects,
    ``(dston, dstoff)`` for zones with an annual DST transition, or
    ``None`` for fixed offset zones.

### Function: _get_hasdst(self)

### Function: _dst_base_offset(self)

### Function: __init__(self, name)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: __init__(self)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __reduce__(self)
