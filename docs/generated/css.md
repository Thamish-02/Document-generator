## AI Summary

A file named css.py.


## Class: CSS

**Description:** A proxy object against the ``soupsieve`` library, to simplify its
CSS selector API.

You don't need to instantiate this class yourself; instead, use
`element.Tag.css`.

:param tag: All CSS selectors run by this object will use this as
    their starting point.

:param api: An optional drop-in replacement for the ``soupsieve`` module,
    intended for use in unit tests.

### Function: __init__(self, tag, api)

### Function: escape(self, ident)

**Description:** Escape a CSS identifier.

This is a simple wrapper around `soupsieve.escape() <https://facelessuser.github.io/soupsieve/api/#soupsieveescape>`_. See the
documentation for that function for more information.

### Function: _ns(self, ns, select)

**Description:** Normalize a dictionary of namespaces.

### Function: _rs(self, results)

**Description:** Normalize a list of results to a py:class:`ResultSet`.

A py:class:`ResultSet` is more consistent with the rest of
Beautiful Soup's API, and :py:meth:`ResultSet.__getattr__` has
a helpful error message if you try to treat a list of results
as a single result (a common mistake).

### Function: compile(self, select, namespaces, flags)

**Description:** Pre-compile a selector and return the compiled object.

:param selector: A CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
   used in the CSS selector to namespace URIs. By default,
   Beautiful Soup will use the prefixes it encountered while
   parsing the document.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.compile() <https://facelessuser.github.io/soupsieve/api/#soupsievecompile>`_ method.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   `soupsieve.compile() <https://facelessuser.github.io/soupsieve/api/#soupsievecompile>`_ method.

:return: A precompiled selector object.
:rtype: soupsieve.SoupSieve

### Function: select_one(self, select, namespaces, flags)

**Description:** Perform a CSS selection operation on the current Tag and return the
first result, if any.

This uses the Soup Sieve library. For more information, see
that library's documentation for the `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.

:param selector: A CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
   used in the CSS selector to namespace URIs. By default,
   Beautiful Soup will use the prefixes it encountered while
   parsing the document.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.

### Function: select(self, select, namespaces, limit, flags)

**Description:** Perform a CSS selection operation on the current `element.Tag`.

This uses the Soup Sieve library. For more information, see
that library's documentation for the `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.

:param selector: A CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
    used in the CSS selector to namespace URIs. By default,
    Beautiful Soup will pass in the prefixes it encountered while
    parsing the document.

:param limit: After finding this number of results, stop looking.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.

### Function: iselect(self, select, namespaces, limit, flags)

**Description:** Perform a CSS selection operation on the current `element.Tag`.

This uses the Soup Sieve library. For more information, see
that library's documentation for the `soupsieve.iselect()
<https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_
method. It is the same as select(), but it returns a generator
instead of a list.

:param selector: A string containing a CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
    used in the CSS selector to namespace URIs. By default,
    Beautiful Soup will pass in the prefixes it encountered while
    parsing the document.

:param limit: After finding this number of results, stop looking.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.iselect() <https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_ method.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   `soupsieve.iselect() <https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_ method.

### Function: closest(self, select, namespaces, flags)

**Description:** Find the `element.Tag` closest to this one that matches the given selector.

This uses the Soup Sieve library. For more information, see
that library's documentation for the `soupsieve.closest()
<https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_
method.

:param selector: A string containing a CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
    used in the CSS selector to namespace URIs. By default,
    Beautiful Soup will pass in the prefixes it encountered while
    parsing the document.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.closest() <https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_ method.

:param kwargs: Keyword arguments to be passed into Soup Sieve's
   `soupsieve.closest() <https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_ method.

### Function: match(self, select, namespaces, flags)

**Description:** Check whether or not this `element.Tag` matches the given CSS selector.

This uses the Soup Sieve library. For more information, see
that library's documentation for the `soupsieve.match()
<https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
method.

:param: a CSS selector.

:param namespaces: A dictionary mapping namespace prefixes
    used in the CSS selector to namespace URIs. By default,
    Beautiful Soup will pass in the prefixes it encountered while
    parsing the document.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.match()
    <https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
    method.

:param kwargs: Keyword arguments to be passed into SoupSieve's
    `soupsieve.match()
    <https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
    method.

### Function: filter(self, select, namespaces, flags)

**Description:** Filter this `element.Tag`'s direct children based on the given CSS selector.

This uses the Soup Sieve library. It works the same way as
passing a `element.Tag` into that library's `soupsieve.filter()
<https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
method. For more information, see the documentation for
`soupsieve.filter()
<https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_.

:param namespaces: A dictionary mapping namespace prefixes
    used in the CSS selector to namespace URIs. By default,
    Beautiful Soup will pass in the prefixes it encountered while
    parsing the document.

:param flags: Flags to be passed into Soup Sieve's
    `soupsieve.filter()
    <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
    method.

:param kwargs: Keyword arguments to be passed into SoupSieve's
    `soupsieve.filter()
    <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
    method.
