## AI Summary

A file named _pydev_getopt.py.


## Class: GetoptError

### Function: gnu_getopt(args, shortopts, longopts)

**Description:** getopt(args, options[, long_options]) -> opts, args

This function works like getopt(), except that GNU style scanning
mode is used by default. This means that option and non-option
arguments may be intermixed. The getopt() function stops
processing options as soon as a non-option argument is
encountered.

If the first character of the option string is `+', or if the
environment variable POSIXLY_CORRECT is set, then option
processing stops as soon as a non-option argument is encountered.

### Function: do_longs(opts, opt, longopts, args)

### Function: long_has_args(opt, longopts)

### Function: do_shorts(opts, optstring, shortopts, args)

### Function: short_has_arg(opt, shortopts)

### Function: __init__(self, msg, opt)

### Function: __str__(self)
