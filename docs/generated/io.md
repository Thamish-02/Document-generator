## AI Summary

A file named io.py.


### Function: unicode_std_stream(stream)

**Description:** Get a wrapper to write unicode to stdout/stderr as UTF-8.

This ignores environment variables and default encodings, to reliably write
unicode to stdout or stderr.

::

    unicode_std_stream().write(u'ł@e¶ŧ←')

### Function: unicode_stdin_stream()

**Description:** Get a wrapper to read unicode from stdin as UTF-8.

This ignores environment variables and default encodings, to reliably read unicode from stdin.

::

    totreat = unicode_stdin_stream().read()

## Class: FormatSafeDict

**Description:** Format a dictionary safely.

### Function: link(src, dst)

**Description:** Hard links ``src`` to ``dst``, returning 0 or errno.

Note that the special errno ``ENOLINK`` will be returned if ``os.link`` isn't
supported by the operating system.

### Function: link_or_copy(src, dst)

**Description:** Attempts to hardlink ``src`` to ``dst``, copying if the link fails.

Attempts to maintain the semantics of ``shutil.copy``.

Because ``os.link`` does not overwrite files, a unique temporary file
will be used if the target already exists, then that file will be moved
into place.

### Function: __missing__(self, key)

**Description:** Handle missing value.
