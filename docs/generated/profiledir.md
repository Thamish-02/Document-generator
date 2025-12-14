## AI Summary

A file named profiledir.py.


## Class: ProfileDirError

## Class: ProfileDir

**Description:** An object to manage the profile directory and its resources.

The profile directory is used by all IPython applications, to manage
configuration, logging and security.

This object knows how to find, create and manage these directories. This
should be used by any code that wants to handle profiles.

### Function: _location_changed(self, change)

### Function: _mkdir(self, path, mode)

**Description:** ensure a directory exists at a given path

This is a version of os.mkdir, with the following differences:

- returns whether the directory has been created or not.
- ignores EEXIST, protecting against race conditions where
  the dir may have been created in between the check and
  the creation
- sets permissions if requested and the dir already exists

Parameters
----------
path: str
    path of the dir to create
mode: int
    see `mode` of `os.mkdir`

Returns
-------
bool:
    returns True if it created the directory, False otherwise

### Function: check_log_dir(self, change)

### Function: check_startup_dir(self, change)

### Function: check_security_dir(self, change)

### Function: check_pid_dir(self, change)

### Function: check_dirs(self)

### Function: copy_config_file(self, config_file, path, overwrite)

**Description:** Copy a default config file into the active profile directory.

Default configuration files are kept in :mod:`IPython.core.profile`.
This function moves these from that location to the working profile
directory.

### Function: create_profile_dir(cls, profile_dir, config)

**Description:** Create a new profile directory given a full path.

Parameters
----------
profile_dir : str
    The full path to the profile directory.  If it does exist, it will
    be used.  If not, it will be created.

### Function: create_profile_dir_by_name(cls, path, name, config)

**Description:** Create a profile dir by profile name and path.

Parameters
----------
path : unicode
    The path (directory) to put the profile directory in.
name : unicode
    The name of the profile.  The name of the profile directory will
    be "profile_<profile>".

### Function: find_profile_dir_by_name(cls, ipython_dir, name, config)

**Description:** Find an existing profile dir by profile name, return its ProfileDir.

This searches through a sequence of paths for a profile dir.  If it
is not found, a :class:`ProfileDirError` exception will be raised.

The search path algorithm is:
1. ``os.getcwd()`` # removed for security reason.
2. ``ipython_dir``

Parameters
----------
ipython_dir : unicode or str
    The IPython directory to use.
name : unicode or str
    The name of the profile.  The name of the profile directory
    will be "profile_<profile>".

### Function: find_profile_dir(cls, profile_dir, config)

**Description:** Find/create a profile dir and return its ProfileDir.

This will create the profile directory if it doesn't exist.

Parameters
----------
profile_dir : unicode or str
    The path of the profile directory.
