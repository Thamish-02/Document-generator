## AI Summary

A file named localedata.py.


### Function: normalize_locale(name)

**Description:** Normalize a locale ID by stripping spaces and apply proper casing.

Returns the normalized locale ID string or `None` if the ID is not
recognized.

### Function: resolve_locale_filename(name)

**Description:** Resolve a locale identifier to a `.dat` path on disk.

### Function: exists(name)

**Description:** Check whether locale data is available for the given locale.

Returns `True` if it exists, `False` otherwise.

:param name: the locale identifier string

### Function: locale_identifiers()

**Description:** Return a list of all locale identifiers for which locale data is
available.

This data is cached after the first invocation.
You can clear the cache by calling `locale_identifiers.cache_clear()`.

.. versionadded:: 0.8.1

:return: a list of locale identifiers (strings)

### Function: _is_non_likely_script(name)

**Description:** Return whether the locale is of the form ``lang_Script``,
and the script is not the likely script for the language.

This implements the behavior of the ``nonlikelyScript`` value of the
``localRules`` attribute for parent locales added in CLDR 45.

### Function: load(name, merge_inherited)

**Description:** Load the locale data for the given locale.

The locale data is a dictionary that contains much of the data defined by
the Common Locale Data Repository (CLDR). This data is stored as a
collection of pickle files inside the ``babel`` package.

>>> d = load('en_US')
>>> d['languages']['sv']
u'Swedish'

Note that the results are cached, and subsequent requests for the same
locale return the same dictionary:

>>> d1 = load('en_US')
>>> d2 = load('en_US')
>>> d1 is d2
True

:param name: the locale identifier string (or "root")
:param merge_inherited: whether the inherited data should be merged into
                        the data of the requested locale
:raise `IOError`: if no locale data file is found for the given locale
                  identifier, or one of the locales it inherits from

### Function: merge(dict1, dict2)

**Description:** Merge the data from `dict2` into the `dict1` dictionary, making copies
of nested dictionaries.

>>> d = {1: 'foo', 3: 'baz'}
>>> merge(d, {1: 'Foo', 2: 'Bar'})
>>> sorted(d.items())
[(1, 'Foo'), (2, 'Bar'), (3, 'baz')]

:param dict1: the dictionary to merge into
:param dict2: the dictionary containing the data that should be merged

## Class: Alias

**Description:** Representation of an alias in the locale data.

An alias is a value that refers to some other part of the locale data,
as specified by the `keys`.

## Class: LocaleDataDict

**Description:** Dictionary wrapper that automatically resolves aliases to the actual
values.

### Function: __init__(self, keys)

### Function: __repr__(self)

### Function: resolve(self, data)

**Description:** Resolve the alias based on the given data.

This is done recursively, so if one alias resolves to a second alias,
that second alias will also be resolved.

:param data: the locale data
:type data: `dict`

### Function: __init__(self, data, base)

### Function: __len__(self)

### Function: __iter__(self)

### Function: __getitem__(self, key)

### Function: __setitem__(self, key, value)

### Function: __delitem__(self, key)

### Function: copy(self)
