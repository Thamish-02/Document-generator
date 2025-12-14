## AI Summary

A file named distro.py.


## Class: VersionDict

## Class: InfoDict

### Function: linux_distribution(full_distribution_name)

**Description:** .. deprecated:: 1.6.0

    :func:`distro.linux_distribution()` is deprecated. It should only be
    used as a compatibility shim with Python's
    :py:func:`platform.linux_distribution()`. Please use :func:`distro.id`,
    :func:`distro.version` and :func:`distro.name` instead.

Return information about the current OS distribution as a tuple
``(id_name, version, codename)`` with items as follows:

* ``id_name``:  If *full_distribution_name* is false, the result of
  :func:`distro.id`. Otherwise, the result of :func:`distro.name`.

* ``version``:  The result of :func:`distro.version`.

* ``codename``:  The extra item (usually in parentheses) after the
  os-release version number, or the result of :func:`distro.codename`.

The interface of this function is compatible with the original
:py:func:`platform.linux_distribution` function, supporting a subset of
its parameters.

The data it returns may not exactly be the same, because it uses more data
sources than the original function, and that may lead to different data if
the OS distribution is not consistent across multiple data sources it
provides (there are indeed such distributions ...).

Another reason for differences is the fact that the :func:`distro.id`
method normalizes the distro ID string to a reliable machine-readable value
for a number of popular OS distributions.

### Function: id()

**Description:** Return the distro ID of the current distribution, as a
machine-readable string.

For a number of OS distributions, the returned distro ID value is
*reliable*, in the sense that it is documented and that it does not change
across releases of the distribution.

This package maintains the following reliable distro ID values:

==============  =========================================
Distro ID       Distribution
==============  =========================================
"ubuntu"        Ubuntu
"debian"        Debian
"rhel"          RedHat Enterprise Linux
"centos"        CentOS
"fedora"        Fedora
"sles"          SUSE Linux Enterprise Server
"opensuse"      openSUSE
"amzn"          Amazon Linux
"arch"          Arch Linux
"buildroot"     Buildroot
"cloudlinux"    CloudLinux OS
"exherbo"       Exherbo Linux
"gentoo"        GenToo Linux
"ibm_powerkvm"  IBM PowerKVM
"kvmibm"        KVM for IBM z Systems
"linuxmint"     Linux Mint
"mageia"        Mageia
"mandriva"      Mandriva Linux
"parallels"     Parallels
"pidora"        Pidora
"raspbian"      Raspbian
"oracle"        Oracle Linux (and Oracle Enterprise Linux)
"scientific"    Scientific Linux
"slackware"     Slackware
"xenserver"     XenServer
"openbsd"       OpenBSD
"netbsd"        NetBSD
"freebsd"       FreeBSD
"midnightbsd"   MidnightBSD
"rocky"         Rocky Linux
"aix"           AIX
"guix"          Guix System
"altlinux"      ALT Linux
==============  =========================================

If you have a need to get distros for reliable IDs added into this set,
or if you find that the :func:`distro.id` function returns a different
distro ID for one of the listed distros, please create an issue in the
`distro issue tracker`_.

**Lookup hierarchy and transformations:**

First, the ID is obtained from the following sources, in the specified
order. The first available and non-empty value is used:

* the value of the "ID" attribute of the os-release file,

* the value of the "Distributor ID" attribute returned by the lsb_release
  command,

* the first part of the file name of the distro release file,

The so determined ID value then passes the following transformations,
before it is returned by this method:

* it is translated to lower case,

* blanks (which should not be there anyway) are translated to underscores,

* a normalization of the ID is performed, based upon
  `normalization tables`_. The purpose of this normalization is to ensure
  that the ID is as reliable as possible, even across incompatible changes
  in the OS distributions. A common reason for an incompatible change is
  the addition of an os-release file, or the addition of the lsb_release
  command, with ID values that differ from what was previously determined
  from the distro release file name.

### Function: name(pretty)

**Description:** Return the name of the current OS distribution, as a human-readable
string.

If *pretty* is false, the name is returned without version or codename.
(e.g. "CentOS Linux")

If *pretty* is true, the version and codename are appended.
(e.g. "CentOS Linux 7.1.1503 (Core)")

**Lookup hierarchy:**

The name is obtained from the following sources, in the specified order.
The first available and non-empty value is used:

* If *pretty* is false:

  - the value of the "NAME" attribute of the os-release file,

  - the value of the "Distributor ID" attribute returned by the lsb_release
    command,

  - the value of the "<name>" field of the distro release file.

* If *pretty* is true:

  - the value of the "PRETTY_NAME" attribute of the os-release file,

  - the value of the "Description" attribute returned by the lsb_release
    command,

  - the value of the "<name>" field of the distro release file, appended
    with the value of the pretty version ("<version_id>" and "<codename>"
    fields) of the distro release file, if available.

### Function: version(pretty, best)

**Description:** Return the version of the current OS distribution, as a human-readable
string.

If *pretty* is false, the version is returned without codename (e.g.
"7.0").

If *pretty* is true, the codename in parenthesis is appended, if the
codename is non-empty (e.g. "7.0 (Maipo)").

Some distributions provide version numbers with different precisions in
the different sources of distribution information. Examining the different
sources in a fixed priority order does not always yield the most precise
version (e.g. for Debian 8.2, or CentOS 7.1).

Some other distributions may not provide this kind of information. In these
cases, an empty string would be returned. This behavior can be observed
with rolling releases distributions (e.g. Arch Linux).

The *best* parameter can be used to control the approach for the returned
version:

If *best* is false, the first non-empty version number in priority order of
the examined sources is returned.

If *best* is true, the most precise version number out of all examined
sources is returned.

**Lookup hierarchy:**

In all cases, the version number is obtained from the following sources.
If *best* is false, this order represents the priority order:

* the value of the "VERSION_ID" attribute of the os-release file,
* the value of the "Release" attribute returned by the lsb_release
  command,
* the version number parsed from the "<version_id>" field of the first line
  of the distro release file,
* the version number parsed from the "PRETTY_NAME" attribute of the
  os-release file, if it follows the format of the distro release files.
* the version number parsed from the "Description" attribute returned by
  the lsb_release command, if it follows the format of the distro release
  files.

### Function: version_parts(best)

**Description:** Return the version of the current OS distribution as a tuple
``(major, minor, build_number)`` with items as follows:

* ``major``:  The result of :func:`distro.major_version`.

* ``minor``:  The result of :func:`distro.minor_version`.

* ``build_number``:  The result of :func:`distro.build_number`.

For a description of the *best* parameter, see the :func:`distro.version`
method.

### Function: major_version(best)

**Description:** Return the major version of the current OS distribution, as a string,
if provided.
Otherwise, the empty string is returned. The major version is the first
part of the dot-separated version string.

For a description of the *best* parameter, see the :func:`distro.version`
method.

### Function: minor_version(best)

**Description:** Return the minor version of the current OS distribution, as a string,
if provided.
Otherwise, the empty string is returned. The minor version is the second
part of the dot-separated version string.

For a description of the *best* parameter, see the :func:`distro.version`
method.

### Function: build_number(best)

**Description:** Return the build number of the current OS distribution, as a string,
if provided.
Otherwise, the empty string is returned. The build number is the third part
of the dot-separated version string.

For a description of the *best* parameter, see the :func:`distro.version`
method.

### Function: like()

**Description:** Return a space-separated list of distro IDs of distributions that are
closely related to the current OS distribution in regards to packaging
and programming interfaces, for example distributions the current
distribution is a derivative from.

**Lookup hierarchy:**

This information item is only provided by the os-release file.
For details, see the description of the "ID_LIKE" attribute in the
`os-release man page
<http://www.freedesktop.org/software/systemd/man/os-release.html>`_.

### Function: codename()

**Description:** Return the codename for the release of the current OS distribution,
as a string.

If the distribution does not have a codename, an empty string is returned.

Note that the returned codename is not always really a codename. For
example, openSUSE returns "x86_64". This function does not handle such
cases in any special way and just returns the string it finds, if any.

**Lookup hierarchy:**

* the codename within the "VERSION" attribute of the os-release file, if
  provided,

* the value of the "Codename" attribute returned by the lsb_release
  command,

* the value of the "<codename>" field of the distro release file.

### Function: info(pretty, best)

**Description:** Return certain machine-readable information items about the current OS
distribution in a dictionary, as shown in the following example:

.. sourcecode:: python

    {
        'id': 'rhel',
        'version': '7.0',
        'version_parts': {
            'major': '7',
            'minor': '0',
            'build_number': ''
        },
        'like': 'fedora',
        'codename': 'Maipo'
    }

The dictionary structure and keys are always the same, regardless of which
information items are available in the underlying data sources. The values
for the various keys are as follows:

* ``id``:  The result of :func:`distro.id`.

* ``version``:  The result of :func:`distro.version`.

* ``version_parts -> major``:  The result of :func:`distro.major_version`.

* ``version_parts -> minor``:  The result of :func:`distro.minor_version`.

* ``version_parts -> build_number``:  The result of
  :func:`distro.build_number`.

* ``like``:  The result of :func:`distro.like`.

* ``codename``:  The result of :func:`distro.codename`.

For a description of the *pretty* and *best* parameters, see the
:func:`distro.version` method.

### Function: os_release_info()

**Description:** Return a dictionary containing key-value pairs for the information items
from the os-release file data source of the current OS distribution.

See `os-release file`_ for details about these information items.

### Function: lsb_release_info()

**Description:** Return a dictionary containing key-value pairs for the information items
from the lsb_release command data source of the current OS distribution.

See `lsb_release command output`_ for details about these information
items.

### Function: distro_release_info()

**Description:** Return a dictionary containing key-value pairs for the information items
from the distro release file data source of the current OS distribution.

See `distro release file`_ for details about these information items.

### Function: uname_info()

**Description:** Return a dictionary containing key-value pairs for the information items
from the distro release file data source of the current OS distribution.

### Function: os_release_attr(attribute)

**Description:** Return a single named information item from the os-release file data source
of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
  The empty string, if the item does not exist.

See `os-release file`_ for details about these information items.

### Function: lsb_release_attr(attribute)

**Description:** Return a single named information item from the lsb_release command output
data source of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
  The empty string, if the item does not exist.

See `lsb_release command output`_ for details about these information
items.

### Function: distro_release_attr(attribute)

**Description:** Return a single named information item from the distro release file
data source of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
  The empty string, if the item does not exist.

See `distro release file`_ for details about these information items.

### Function: uname_attr(attribute)

**Description:** Return a single named information item from the distro release file
data source of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
            The empty string, if the item does not exist.

## Class: LinuxDistribution

**Description:** Provides information about a OS distribution.

This package creates a private module-global instance of this class with
default initialization arguments, that is used by the
`consolidated accessor functions`_ and `single source accessor functions`_.
By using default initialization arguments, that module-global instance
returns data about the current OS distribution (i.e. the distro this
package runs on).

Normally, it is not necessary to create additional instances of this class.
However, in situations where control is needed over the exact data sources
that are used, instances of this class can be created with a specific
distro release file, or a specific os-release file, or without invoking the
lsb_release command.

### Function: main()

### Function: __init__(self, include_lsb, os_release_file, distro_release_file, include_uname, root_dir, include_oslevel)

**Description:** The initialization method of this class gathers information from the
available data sources, and stores that in private instance attributes.
Subsequent access to the information items uses these private instance
attributes, so that the data sources are read only once.

Parameters:

* ``include_lsb`` (bool): Controls whether the
  `lsb_release command output`_ is included as a data source.

  If the lsb_release command is not available in the program execution
  path, the data source for the lsb_release command will be empty.

* ``os_release_file`` (string): The path name of the
  `os-release file`_ that is to be used as a data source.

  An empty string (the default) will cause the default path name to
  be used (see `os-release file`_ for details).

  If the specified or defaulted os-release file does not exist, the
  data source for the os-release file will be empty.

* ``distro_release_file`` (string): The path name of the
  `distro release file`_ that is to be used as a data source.

  An empty string (the default) will cause a default search algorithm
  to be used (see `distro release file`_ for details).

  If the specified distro release file does not exist, or if no default
  distro release file can be found, the data source for the distro
  release file will be empty.

* ``include_uname`` (bool): Controls whether uname command output is
  included as a data source. If the uname command is not available in
  the program execution path the data source for the uname command will
  be empty.

* ``root_dir`` (string): The absolute path to the root directory to use
  to find distro-related information files. Note that ``include_*``
  parameters must not be enabled in combination with ``root_dir``.

* ``include_oslevel`` (bool): Controls whether (AIX) oslevel command
  output is included as a data source. If the oslevel command is not
  available in the program execution path the data source will be
  empty.

Public instance attributes:

* ``os_release_file`` (string): The path name of the
  `os-release file`_ that is actually used as a data source. The
  empty string if no distro release file is used as a data source.

* ``distro_release_file`` (string): The path name of the
  `distro release file`_ that is actually used as a data source. The
  empty string if no distro release file is used as a data source.

* ``include_lsb`` (bool): The result of the ``include_lsb`` parameter.
  This controls whether the lsb information will be loaded.

* ``include_uname`` (bool): The result of the ``include_uname``
  parameter. This controls whether the uname information will
  be loaded.

* ``include_oslevel`` (bool): The result of the ``include_oslevel``
  parameter. This controls whether (AIX) oslevel information will be
  loaded.

* ``root_dir`` (string): The result of the ``root_dir`` parameter.
  The absolute path to the root directory to use to find distro-related
  information files.

Raises:

* :py:exc:`ValueError`: Initialization parameters combination is not
   supported.

* :py:exc:`OSError`: Some I/O issue with an os-release file or distro
  release file.

* :py:exc:`UnicodeError`: A data source has unexpected characters or
  uses an unexpected encoding.

### Function: __repr__(self)

**Description:** Return repr of all info

### Function: linux_distribution(self, full_distribution_name)

**Description:** Return information about the OS distribution that is compatible
with Python's :func:`platform.linux_distribution`, supporting a subset
of its parameters.

For details, see :func:`distro.linux_distribution`.

### Function: id(self)

**Description:** Return the distro ID of the OS distribution, as a string.

For details, see :func:`distro.id`.

### Function: name(self, pretty)

**Description:** Return the name of the OS distribution, as a string.

For details, see :func:`distro.name`.

### Function: version(self, pretty, best)

**Description:** Return the version of the OS distribution, as a string.

For details, see :func:`distro.version`.

### Function: version_parts(self, best)

**Description:** Return the version of the OS distribution, as a tuple of version
numbers.

For details, see :func:`distro.version_parts`.

### Function: major_version(self, best)

**Description:** Return the major version number of the current distribution.

For details, see :func:`distro.major_version`.

### Function: minor_version(self, best)

**Description:** Return the minor version number of the current distribution.

For details, see :func:`distro.minor_version`.

### Function: build_number(self, best)

**Description:** Return the build number of the current distribution.

For details, see :func:`distro.build_number`.

### Function: like(self)

**Description:** Return the IDs of distributions that are like the OS distribution.

For details, see :func:`distro.like`.

### Function: codename(self)

**Description:** Return the codename of the OS distribution.

For details, see :func:`distro.codename`.

### Function: info(self, pretty, best)

**Description:** Return certain machine-readable information about the OS
distribution.

For details, see :func:`distro.info`.

### Function: os_release_info(self)

**Description:** Return a dictionary containing key-value pairs for the information
items from the os-release file data source of the OS distribution.

For details, see :func:`distro.os_release_info`.

### Function: lsb_release_info(self)

**Description:** Return a dictionary containing key-value pairs for the information
items from the lsb_release command data source of the OS
distribution.

For details, see :func:`distro.lsb_release_info`.

### Function: distro_release_info(self)

**Description:** Return a dictionary containing key-value pairs for the information
items from the distro release file data source of the OS
distribution.

For details, see :func:`distro.distro_release_info`.

### Function: uname_info(self)

**Description:** Return a dictionary containing key-value pairs for the information
items from the uname command data source of the OS distribution.

For details, see :func:`distro.uname_info`.

### Function: oslevel_info(self)

**Description:** Return AIX' oslevel command output.

### Function: os_release_attr(self, attribute)

**Description:** Return a single named information item from the os-release file data
source of the OS distribution.

For details, see :func:`distro.os_release_attr`.

### Function: lsb_release_attr(self, attribute)

**Description:** Return a single named information item from the lsb_release command
output data source of the OS distribution.

For details, see :func:`distro.lsb_release_attr`.

### Function: distro_release_attr(self, attribute)

**Description:** Return a single named information item from the distro release file
data source of the OS distribution.

For details, see :func:`distro.distro_release_attr`.

### Function: uname_attr(self, attribute)

**Description:** Return a single named information item from the uname command
output data source of the OS distribution.

For details, see :func:`distro.uname_attr`.

### Function: _os_release_info(self)

**Description:** Get the information items from the specified os-release file.

Returns:
    A dictionary containing all information items.

### Function: _parse_os_release_content(lines)

**Description:** Parse the lines of an os-release file.

Parameters:

* lines: Iterable through the lines in the os-release file.
         Each line must be a unicode string or a UTF-8 encoded byte
         string.

Returns:
    A dictionary containing all information items.

### Function: _lsb_release_info(self)

**Description:** Get the information items from the lsb_release command output.

Returns:
    A dictionary containing all information items.

### Function: _parse_lsb_release_content(lines)

**Description:** Parse the output of the lsb_release command.

Parameters:

* lines: Iterable through the lines of the lsb_release output.
         Each line must be a unicode string or a UTF-8 encoded byte
         string.

Returns:
    A dictionary containing all information items.

### Function: _uname_info(self)

### Function: _oslevel_info(self)

### Function: _debian_version(self)

### Function: _parse_uname_content(lines)

### Function: _to_str(bytestring)

### Function: _distro_release_info(self)

**Description:** Get the information items from the specified distro release file.

Returns:
    A dictionary containing all information items.

### Function: _parse_distro_release_file(self, filepath)

**Description:** Parse a distro release file.

Parameters:

* filepath: Path name of the distro release file.

Returns:
    A dictionary containing all information items.

### Function: _parse_distro_release_content(line)

**Description:** Parse a line from a distro release file.

Parameters:
* line: Line from the distro release file. Must be a unicode string
        or a UTF-8 encoded byte string.

Returns:
    A dictionary containing all information items.

## Class: cached_property

**Description:** A version of @property which caches the value.  On access, it calls the
underlying function and sets the value in `__dict__` so future accesses
will not re-call the property.

### Function: normalize(distro_id, table)

### Function: __init__(self, f)

### Function: __get__(self, obj, owner)
