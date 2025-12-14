## AI Summary

A file named bccache.py.


## Class: Bucket

**Description:** Buckets are used to store the bytecode for one template.  It's created
and initialized by the bytecode cache and passed to the loading functions.

The buckets get an internal checksum from the cache assigned and use this
to automatically reject outdated cache material.  Individual bytecode
cache subclasses don't have to care about cache invalidation.

## Class: BytecodeCache

**Description:** To implement your own bytecode cache you have to subclass this class
and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
these methods are passed a :class:`~jinja2.bccache.Bucket`.

A very basic bytecode cache that saves the bytecode on the file system::

    from os import path

    class MyCache(BytecodeCache):

        def __init__(self, directory):
            self.directory = directory

        def load_bytecode(self, bucket):
            filename = path.join(self.directory, bucket.key)
            if path.exists(filename):
                with open(filename, 'rb') as f:
                    bucket.load_bytecode(f)

        def dump_bytecode(self, bucket):
            filename = path.join(self.directory, bucket.key)
            with open(filename, 'wb') as f:
                bucket.write_bytecode(f)

A more advanced version of a filesystem based bytecode cache is part of
Jinja.

## Class: FileSystemBytecodeCache

**Description:** A bytecode cache that stores bytecode on the filesystem.  It accepts
two arguments: The directory where the cache items are stored and a
pattern string that is used to build the filename.

If no directory is specified a default cache directory is selected.  On
Windows the user's temp directory is used, on UNIX systems a directory
is created for the user in the system temp directory.

The pattern can be used to have multiple separate caches operate on the
same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
is replaced with the cache key.

>>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

This bytecode cache supports clearing of the cache using the clear method.

## Class: MemcachedBytecodeCache

**Description:** This class implements a bytecode cache that uses a memcache cache for
storing the information.  It does not enforce a specific memcache library
(tummy's memcache or cmemcache) but will accept any class that provides
the minimal interface required.

Libraries compatible with this class:

-   `cachelib <https://github.com/pallets/cachelib>`_
-   `python-memcached <https://pypi.org/project/python-memcached/>`_

(Unfortunately the django cache interface is not compatible because it
does not support storing binary data, only text. You can however pass
the underlying cache client to the bytecode cache which is available
as `django.core.cache.cache._client`.)

The minimal interface for the client passed to the constructor is this:

.. class:: MinimalClientInterface

    .. method:: set(key, value[, timeout])

        Stores the bytecode in the cache.  `value` is a string and
        `timeout` the timeout of the key.  If timeout is not provided
        a default timeout or no timeout should be assumed, if it's
        provided it's an integer with the number of seconds the cache
        item should exist.

    .. method:: get(key)

        Returns the value for the cache key.  If the item does not
        exist in the cache the return value must be `None`.

The other arguments to the constructor are the prefix for all keys that
is added before the actual cache key and the timeout for the bytecode in
the cache system.  We recommend a high (or no) timeout.

This bytecode cache does not support clearing of used items in the cache.
The clear method is a no-operation function.

.. versionadded:: 2.7
   Added support for ignoring memcache errors through the
   `ignore_memcache_errors` parameter.

## Class: _MemcachedClient

### Function: __init__(self, environment, key, checksum)

### Function: reset(self)

**Description:** Resets the bucket (unloads the bytecode).

### Function: load_bytecode(self, f)

**Description:** Loads bytecode from a file or file like object.

### Function: write_bytecode(self, f)

**Description:** Dump the bytecode into the file or file like object passed.

### Function: bytecode_from_string(self, string)

**Description:** Load bytecode from bytes.

### Function: bytecode_to_string(self)

**Description:** Return the bytecode as bytes.

### Function: load_bytecode(self, bucket)

**Description:** Subclasses have to override this method to load bytecode into a
bucket.  If they are not able to find code in the cache for the
bucket, it must not do anything.

### Function: dump_bytecode(self, bucket)

**Description:** Subclasses have to override this method to write the bytecode
from a bucket back to the cache.  If it unable to do so it must not
fail silently but raise an exception.

### Function: clear(self)

**Description:** Clears the cache.  This method is not used by Jinja but should be
implemented to allow applications to clear the bytecode cache used
by a particular environment.

### Function: get_cache_key(self, name, filename)

**Description:** Returns the unique hash key for this template name.

### Function: get_source_checksum(self, source)

**Description:** Returns a checksum for the source.

### Function: get_bucket(self, environment, name, filename, source)

**Description:** Return a cache bucket for the given template.  All arguments are
mandatory but filename may be `None`.

### Function: set_bucket(self, bucket)

**Description:** Put the bucket into the cache.

### Function: __init__(self, directory, pattern)

### Function: _get_default_cache_dir(self)

### Function: _get_cache_filename(self, bucket)

### Function: load_bytecode(self, bucket)

### Function: dump_bytecode(self, bucket)

### Function: clear(self)

### Function: __init__(self, client, prefix, timeout, ignore_memcache_errors)

### Function: load_bytecode(self, bucket)

### Function: dump_bytecode(self, bucket)

### Function: get(self, key)

### Function: set(self, key, value, timeout)

### Function: _unsafe_dir()

### Function: remove_silent()
