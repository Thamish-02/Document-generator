## AI Summary

A file named _unix.py.


### Function: _tz_from_env(tzenv)

### Function: _get_localzone(_root)

**Description:** Tries to find the local timezone configuration.
This method prefers finding the timezone name and passing that to
zoneinfo or pytz, over passing in the localtime file, as in the later
case the zoneinfo name is unknown.
The parameter _root makes the function look for files like /etc/localtime
beneath the _root directory. This is primarily used by the tests.
In normal usage you call the function without parameters.
