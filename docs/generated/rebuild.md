## AI Summary

A file named rebuild.py.


### Function: rebuild(filename, tag, format, zonegroups, metadata)

**Description:** Rebuild the internal timezone info in dateutil/zoneinfo/zoneinfo*tar*

filename is the timezone tarball from ``ftp.iana.org/tz``.

### Function: _run_zic(zonedir, filepaths)

**Description:** Calls the ``zic`` compiler in a compatible way to get a "fat" binary.

Recent versions of ``zic`` default to ``-b slim``, while older versions
don't even have the ``-b`` option (but default to "fat" binaries). The
current version of dateutil does not support Version 2+ TZif files, which
causes problems when used in conjunction with "slim" binaries, so this
function is used to ensure that we always get a "fat" binary.

### Function: _print_on_nosuchfile(e)

**Description:** Print helpful troubleshooting message

e is an exception raised by subprocess.check_call()
