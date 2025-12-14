## AI Summary

A file named _sysinfo.py.


### Function: pkg_commit_hash(pkg_path)

**Description:** Get short form of commit hash given directory `pkg_path`

We get the commit hash from git if it's a repo.

If this fail, we return a not-found placeholder tuple

Parameters
----------
pkg_path : str
    directory containing package
    only used for getting commit from active repo

Returns
-------
hash_from : str
    Where we got the hash from - description
hash_str : str
    short form of hash

### Function: pkg_info(pkg_path)

**Description:** Return dict describing the context of this package

Parameters
----------
pkg_path : str
    path containing __init__.py for package

Returns
-------
context : dict
    with named parameters of interest

### Function: get_sys_info()

**Description:** Return useful information about the system as a dict.
