## AI Summary

A file named filters.py.


### Function: ignore_case(value)

**Description:** For use as a postprocessor for :func:`make_attrgetter`. Converts strings
to lowercase and returns other types as-is.

### Function: make_attrgetter(environment, attribute, postprocess, default)

**Description:** Returns a callable that looks up the given attribute from a
passed object with the rules of the environment.  Dots are allowed
to access attributes of attributes.  Integer parts in paths are
looked up as integers.

### Function: make_multi_attrgetter(environment, attribute, postprocess)

**Description:** Returns a callable that looks up the given comma separated
attributes from a passed object with the rules of the environment.
Dots are allowed to access attributes of each attribute.  Integer
parts in paths are looked up as integers.

The value returned by the returned callable is a list of extracted
attribute values.

Examples of attribute: "attr1,attr2", "attr1.inner1.0,attr2.inner2.0", etc.

### Function: _prepare_attribute_parts(attr)

### Function: do_forceescape(value)

**Description:** Enforce HTML escaping.  This will probably double escape variables.

### Function: do_urlencode(value)

**Description:** Quote data for use in a URL path or query using UTF-8.

Basic wrapper around :func:`urllib.parse.quote` when given a
string, or :func:`urllib.parse.urlencode` for a dict or iterable.

:param value: Data to quote. A string will be quoted directly. A
    dict or iterable of ``(key, value)`` pairs will be joined as a
    query string.

When given a string, "/" is not quoted. HTTP servers treat "/" and
"%2F" equivalently in paths. If you need quoted slashes, use the
``|replace("/", "%2F")`` filter.

.. versionadded:: 2.7

### Function: do_replace(eval_ctx, s, old, new, count)

**Description:** Return a copy of the value with all occurrences of a substring
replaced with a new one. The first argument is the substring
that should be replaced, the second is the replacement string.
If the optional third argument ``count`` is given, only the first
``count`` occurrences are replaced:

.. sourcecode:: jinja

    {{ "Hello World"|replace("Hello", "Goodbye") }}
        -> Goodbye World

    {{ "aaaaargh"|replace("a", "d'oh, ", 2) }}
        -> d'oh, d'oh, aaargh

### Function: do_upper(s)

**Description:** Convert a value to uppercase.

### Function: do_lower(s)

**Description:** Convert a value to lowercase.

### Function: do_items(value)

**Description:** Return an iterator over the ``(key, value)`` items of a mapping.

``x|items`` is the same as ``x.items()``, except if ``x`` is
undefined an empty iterator is returned.

This filter is useful if you expect the template to be rendered with
an implementation of Jinja in another programming language that does
not have a ``.items()`` method on its mapping type.

.. code-block:: html+jinja

    <dl>
    {% for key, value in my_dict|items %}
        <dt>{{ key }}
        <dd>{{ value }}
    {% endfor %}
    </dl>

.. versionadded:: 3.1

### Function: do_xmlattr(eval_ctx, d, autospace)

**Description:** Create an SGML/XML attribute string based on the items in a dict.

**Values** that are neither ``none`` nor ``undefined`` are automatically
escaped, safely allowing untrusted user input.

User input should not be used as **keys** to this filter. If any key
contains a space, ``/`` solidus, ``>`` greater-than sign, or ``=`` equals
sign, this fails with a ``ValueError``. Regardless of this, user input
should never be used as keys to this filter, or must be separately validated
first.

.. sourcecode:: html+jinja

    <ul{{ {'class': 'my_list', 'missing': none,
            'id': 'list-%d'|format(variable)}|xmlattr }}>
    ...
    </ul>

Results in something like this:

.. sourcecode:: html

    <ul class="my_list" id="list-42">
    ...
    </ul>

As you can see it automatically prepends a space in front of the item
if the filter returned something unless the second parameter is false.

.. versionchanged:: 3.1.4
    Keys with ``/`` solidus, ``>`` greater-than sign, or ``=`` equals sign
    are not allowed.

.. versionchanged:: 3.1.3
    Keys with spaces are not allowed.

### Function: do_capitalize(s)

**Description:** Capitalize a value. The first character will be uppercase, all others
lowercase.

### Function: do_title(s)

**Description:** Return a titlecased version of the value. I.e. words will start with
uppercase letters, all remaining characters are lowercase.

### Function: do_dictsort(value, case_sensitive, by, reverse)

**Description:** Sort a dict and yield (key, value) pairs. Python dicts may not
be in the order you want to display them in, so sort them first.

.. sourcecode:: jinja

    {% for key, value in mydict|dictsort %}
        sort the dict by key, case insensitive

    {% for key, value in mydict|dictsort(reverse=true) %}
        sort the dict by key, case insensitive, reverse order

    {% for key, value in mydict|dictsort(true) %}
        sort the dict by key, case sensitive

    {% for key, value in mydict|dictsort(false, 'value') %}
        sort the dict by value, case insensitive

### Function: do_sort(environment, value, reverse, case_sensitive, attribute)

**Description:** Sort an iterable using Python's :func:`sorted`.

.. sourcecode:: jinja

    {% for city in cities|sort %}
        ...
    {% endfor %}

:param reverse: Sort descending instead of ascending.
:param case_sensitive: When sorting strings, sort upper and lower
    case separately.
:param attribute: When sorting objects or dicts, an attribute or
    key to sort by. Can use dot notation like ``"address.city"``.
    Can be a list of attributes like ``"age,name"``.

The sort is stable, it does not change the relative order of
elements that compare equal. This makes it is possible to chain
sorts on different attributes and ordering.

.. sourcecode:: jinja

    {% for user in users|sort(attribute="name")
        |sort(reverse=true, attribute="age") %}
        ...
    {% endfor %}

As a shortcut to chaining when the direction is the same for all
attributes, pass a comma separate list of attributes.

.. sourcecode:: jinja

    {% for user in users|sort(attribute="age,name") %}
        ...
    {% endfor %}

.. versionchanged:: 2.11.0
    The ``attribute`` parameter can be a comma separated list of
    attributes, e.g. ``"age,name"``.

.. versionchanged:: 2.6
   The ``attribute`` parameter was added.

### Function: sync_do_unique(environment, value, case_sensitive, attribute)

**Description:** Returns a list of unique items from the given iterable.

.. sourcecode:: jinja

    {{ ['foo', 'bar', 'foobar', 'FooBar']|unique|list }}
        -> ['foo', 'bar', 'foobar']

The unique items are yielded in the same order as their first occurrence in
the iterable passed to the filter.

:param case_sensitive: Treat upper and lower case strings as distinct.
:param attribute: Filter objects with unique values for this attribute.

### Function: _min_or_max(environment, value, func, case_sensitive, attribute)

### Function: do_min(environment, value, case_sensitive, attribute)

**Description:** Return the smallest item from the sequence.

.. sourcecode:: jinja

    {{ [1, 2, 3]|min }}
        -> 1

:param case_sensitive: Treat upper and lower case strings as distinct.
:param attribute: Get the object with the min value of this attribute.

### Function: do_max(environment, value, case_sensitive, attribute)

**Description:** Return the largest item from the sequence.

.. sourcecode:: jinja

    {{ [1, 2, 3]|max }}
        -> 3

:param case_sensitive: Treat upper and lower case strings as distinct.
:param attribute: Get the object with the max value of this attribute.

### Function: do_default(value, default_value, boolean)

**Description:** If the value is undefined it will return the passed default value,
otherwise the value of the variable:

.. sourcecode:: jinja

    {{ my_variable|default('my_variable is not defined') }}

This will output the value of ``my_variable`` if the variable was
defined, otherwise ``'my_variable is not defined'``. If you want
to use default with variables that evaluate to false you have to
set the second parameter to `true`:

.. sourcecode:: jinja

    {{ ''|default('the string was empty', true) }}

.. versionchanged:: 2.11
   It's now possible to configure the :class:`~jinja2.Environment` with
   :class:`~jinja2.ChainableUndefined` to make the `default` filter work
   on nested elements and attributes that may contain undefined values
   in the chain without getting an :exc:`~jinja2.UndefinedError`.

### Function: sync_do_join(eval_ctx, value, d, attribute)

**Description:** Return a string which is the concatenation of the strings in the
sequence. The separator between elements is an empty string per
default, you can define it with the optional parameter:

.. sourcecode:: jinja

    {{ [1, 2, 3]|join('|') }}
        -> 1|2|3

    {{ [1, 2, 3]|join }}
        -> 123

It is also possible to join certain attributes of an object:

.. sourcecode:: jinja

    {{ users|join(', ', attribute='username') }}

.. versionadded:: 2.6
   The `attribute` parameter was added.

### Function: do_center(value, width)

**Description:** Centers the value in a field of a given width.

### Function: sync_do_first(environment, seq)

**Description:** Return the first item of a sequence.

### Function: do_last(environment, seq)

**Description:** Return the last item of a sequence.

Note: Does not work with generators. You may want to explicitly
convert it to a list:

.. sourcecode:: jinja

    {{ data | selectattr('name', '==', 'Jinja') | list | last }}

### Function: do_random(context, seq)

**Description:** Return a random item from the sequence.

### Function: do_filesizeformat(value, binary)

**Description:** Format the value like a 'human-readable' file size (i.e. 13 kB,
4.1 MB, 102 Bytes, etc).  Per default decimal prefixes are used (Mega,
Giga, etc.), if the second parameter is set to `True` the binary
prefixes are used (Mebi, Gibi).

### Function: do_pprint(value)

**Description:** Pretty print a variable. Useful for debugging.

### Function: do_urlize(eval_ctx, value, trim_url_limit, nofollow, target, rel, extra_schemes)

**Description:** Convert URLs in text into clickable links.

This may not recognize links in some situations. Usually, a more
comprehensive formatter, such as a Markdown library, is a better
choice.

Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
addresses. Links with trailing punctuation (periods, commas, closing
parentheses) and leading punctuation (opening parentheses) are
recognized excluding the punctuation. Email addresses that include
header fields are not recognized (for example,
``mailto:address@example.com?cc=copy@example.com``).

:param value: Original text containing URLs to link.
:param trim_url_limit: Shorten displayed URL values to this length.
:param nofollow: Add the ``rel=nofollow`` attribute to links.
:param target: Add the ``target`` attribute to links.
:param rel: Add the ``rel`` attribute to links.
:param extra_schemes: Recognize URLs that start with these schemes
    in addition to the default behavior. Defaults to
    ``env.policies["urlize.extra_schemes"]``, which defaults to no
    extra schemes.

.. versionchanged:: 3.0
    The ``extra_schemes`` parameter was added.

.. versionchanged:: 3.0
    Generate ``https://`` links for URLs without a scheme.

.. versionchanged:: 3.0
    The parsing rules were updated. Recognize email addresses with
    or without the ``mailto:`` scheme. Validate IP addresses. Ignore
    parentheses and brackets in more cases.

.. versionchanged:: 2.8
   The ``target`` parameter was added.

### Function: do_indent(s, width, first, blank)

**Description:** Return a copy of the string with each line indented by 4 spaces. The
first line and blank lines are not indented by default.

:param width: Number of spaces, or a string, to indent by.
:param first: Don't skip indenting the first line.
:param blank: Don't skip indenting empty lines.

.. versionchanged:: 3.0
    ``width`` can be a string.

.. versionchanged:: 2.10
    Blank lines are not indented by default.

    Rename the ``indentfirst`` argument to ``first``.

### Function: do_truncate(env, s, length, killwords, end, leeway)

**Description:** Return a truncated copy of the string. The length is specified
with the first parameter which defaults to ``255``. If the second
parameter is ``true`` the filter will cut the text at length. Otherwise
it will discard the last word. If the text was in fact
truncated it will append an ellipsis sign (``"..."``). If you want a
different ellipsis sign than ``"..."`` you can specify it using the
third parameter. Strings that only exceed the length by the tolerance
margin given in the fourth parameter will not be truncated.

.. sourcecode:: jinja

    {{ "foo bar baz qux"|truncate(9) }}
        -> "foo..."
    {{ "foo bar baz qux"|truncate(9, True) }}
        -> "foo ba..."
    {{ "foo bar baz qux"|truncate(11) }}
        -> "foo bar baz qux"
    {{ "foo bar baz qux"|truncate(11, False, '...', 0) }}
        -> "foo bar..."

The default leeway on newer Jinja versions is 5 and was 0 before but
can be reconfigured globally.

### Function: do_wordwrap(environment, s, width, break_long_words, wrapstring, break_on_hyphens)

**Description:** Wrap a string to the given width. Existing newlines are treated
as paragraphs to be wrapped separately.

:param s: Original text to wrap.
:param width: Maximum length of wrapped lines.
:param break_long_words: If a word is longer than ``width``, break
    it across lines.
:param break_on_hyphens: If a word contains hyphens, it may be split
    across lines.
:param wrapstring: String to join each wrapped line. Defaults to
    :attr:`Environment.newline_sequence`.

.. versionchanged:: 2.11
    Existing newlines are treated as paragraphs wrapped separately.

.. versionchanged:: 2.11
    Added the ``break_on_hyphens`` parameter.

.. versionchanged:: 2.7
    Added the ``wrapstring`` parameter.

### Function: do_wordcount(s)

**Description:** Count the words in that string.

### Function: do_int(value, default, base)

**Description:** Convert the value into an integer. If the
conversion doesn't work it will return ``0``. You can
override this default using the first parameter. You
can also override the default base (10) in the second
parameter, which handles input with prefixes such as
0b, 0o and 0x for bases 2, 8 and 16 respectively.
The base is ignored for decimal numbers and non-string values.

### Function: do_float(value, default)

**Description:** Convert the value into a floating point number. If the
conversion doesn't work it will return ``0.0``. You can
override this default using the first parameter.

### Function: do_format(value)

**Description:** Apply the given values to a `printf-style`_ format string, like
``string % values``.

.. sourcecode:: jinja

    {{ "%s, %s!"|format(greeting, name) }}
    Hello, World!

In most cases it should be more convenient and efficient to use the
``%`` operator or :meth:`str.format`.

.. code-block:: text

    {{ "%s, %s!" % (greeting, name) }}
    {{ "{}, {}!".format(greeting, name) }}

.. _printf-style: https://docs.python.org/library/stdtypes.html
    #printf-style-string-formatting

### Function: do_trim(value, chars)

**Description:** Strip leading and trailing characters, by default whitespace.

### Function: do_striptags(value)

**Description:** Strip SGML/XML tags and replace adjacent whitespace by one space.

### Function: sync_do_slice(value, slices, fill_with)

**Description:** Slice an iterator and return a list of lists containing
those items. Useful if you want to create a div containing
three ul tags that represent columns:

.. sourcecode:: html+jinja

    <div class="columnwrapper">
      {%- for column in items|slice(3) %}
        <ul class="column-{{ loop.index }}">
        {%- for item in column %}
          <li>{{ item }}</li>
        {%- endfor %}
        </ul>
      {%- endfor %}
    </div>

If you pass it a second argument it's used to fill missing
values on the last iteration.

### Function: do_batch(value, linecount, fill_with)

**Description:** A filter that batches items. It works pretty much like `slice`
just the other way round. It returns a list of lists with the
given number of items. If you provide a second parameter this
is used to fill up missing items. See this example:

.. sourcecode:: html+jinja

    <table>
    {%- for row in items|batch(3, '&nbsp;') %}
      <tr>
      {%- for column in row %}
        <td>{{ column }}</td>
      {%- endfor %}
      </tr>
    {%- endfor %}
    </table>

### Function: do_round(value, precision, method)

**Description:** Round the number to a given precision. The first
parameter specifies the precision (default is ``0``), the
second the rounding method:

- ``'common'`` rounds either up or down
- ``'ceil'`` always rounds up
- ``'floor'`` always rounds down

If you don't specify a method ``'common'`` is used.

.. sourcecode:: jinja

    {{ 42.55|round }}
        -> 43.0
    {{ 42.55|round(1, 'floor') }}
        -> 42.5

Note that even if rounded to 0 precision, a float is returned.  If
you need a real integer, pipe it through `int`:

.. sourcecode:: jinja

    {{ 42.55|round|int }}
        -> 43

## Class: _GroupTuple

### Function: sync_do_groupby(environment, value, attribute, default, case_sensitive)

**Description:** Group a sequence of objects by an attribute using Python's
:func:`itertools.groupby`. The attribute can use dot notation for
nested access, like ``"address.city"``. Unlike Python's ``groupby``,
the values are sorted first so only one group is returned for each
unique value.

For example, a list of ``User`` objects with a ``city`` attribute
can be rendered in groups. In this example, ``grouper`` refers to
the ``city`` value of the group.

.. sourcecode:: html+jinja

    <ul>{% for city, items in users|groupby("city") %}
      <li>{{ city }}
        <ul>{% for user in items %}
          <li>{{ user.name }}
        {% endfor %}</ul>
      </li>
    {% endfor %}</ul>

``groupby`` yields namedtuples of ``(grouper, list)``, which
can be used instead of the tuple unpacking above. ``grouper`` is the
value of the attribute, and ``list`` is the items with that value.

.. sourcecode:: html+jinja

    <ul>{% for group in users|groupby("city") %}
      <li>{{ group.grouper }}: {{ group.list|join(", ") }}
    {% endfor %}</ul>

You can specify a ``default`` value to use if an object in the list
does not have the given attribute.

.. sourcecode:: jinja

    <ul>{% for city, items in users|groupby("city", default="NY") %}
      <li>{{ city }}: {{ items|map(attribute="name")|join(", ") }}</li>
    {% endfor %}</ul>

Like the :func:`~jinja-filters.sort` filter, sorting and grouping is
case-insensitive by default. The ``key`` for each group will have
the case of the first item in that group of values. For example, if
a list of users has cities ``["CA", "NY", "ca"]``, the "CA" group
will have two values. This can be disabled by passing
``case_sensitive=True``.

.. versionchanged:: 3.1
    Added the ``case_sensitive`` parameter. Sorting and grouping is
    case-insensitive by default, matching other filters that do
    comparisons.

.. versionchanged:: 3.0
    Added the ``default`` parameter.

.. versionchanged:: 2.6
    The attribute supports dot notation for nested access.

### Function: sync_do_sum(environment, iterable, attribute, start)

**Description:** Returns the sum of a sequence of numbers plus the value of parameter
'start' (which defaults to 0).  When the sequence is empty it returns
start.

It is also possible to sum up only certain attributes:

.. sourcecode:: jinja

    Total: {{ items|sum(attribute='price') }}

.. versionchanged:: 2.6
   The ``attribute`` parameter was added to allow summing up over
   attributes.  Also the ``start`` parameter was moved on to the right.

### Function: sync_do_list(value)

**Description:** Convert the value into a list.  If it was a string the returned list
will be a list of characters.

### Function: do_mark_safe(value)

**Description:** Mark the value as safe which means that in an environment with automatic
escaping enabled this variable will not be escaped.

### Function: do_mark_unsafe(value)

**Description:** Mark a value as unsafe.  This is the reverse operation for :func:`safe`.

### Function: do_reverse(value)

### Function: do_reverse(value)

### Function: do_reverse(value)

**Description:** Reverse the object or return an iterator that iterates over it the other
way round.

### Function: do_attr(environment, obj, name)

**Description:** Get an attribute of an object. ``foo|attr("bar")`` works like
``foo.bar``, but returns undefined instead of falling back to ``foo["bar"]``
if the attribute doesn't exist.

See :ref:`Notes on subscriptions <notes-on-subscriptions>` for more details.

### Function: sync_do_map(context, value, name)

### Function: sync_do_map(context, value)

### Function: sync_do_map(context, value)

**Description:** Applies a filter on a sequence of objects or looks up an attribute.
This is useful when dealing with lists of objects but you are really
only interested in a certain value of it.

The basic usage is mapping on an attribute.  Imagine you have a list
of users but you are only interested in a list of usernames:

.. sourcecode:: jinja

    Users on this page: {{ users|map(attribute='username')|join(', ') }}

You can specify a ``default`` value to use if an object in the list
does not have the given attribute.

.. sourcecode:: jinja

    {{ users|map(attribute="username", default="Anonymous")|join(", ") }}

Alternatively you can let it invoke a filter by passing the name of the
filter and the arguments afterwards.  A good example would be applying a
text conversion filter on a sequence:

.. sourcecode:: jinja

    Users on this page: {{ titles|map('lower')|join(', ') }}

Similar to a generator comprehension such as:

.. code-block:: python

    (u.username for u in users)
    (getattr(u, "username", "Anonymous") for u in users)
    (do_lower(x) for x in titles)

.. versionchanged:: 2.11.0
    Added the ``default`` parameter.

.. versionadded:: 2.7

### Function: do_map(context, value, name)

### Function: do_map(context, value)

### Function: sync_do_select(context, value)

**Description:** Filters a sequence of objects by applying a test to each object,
and only selecting the objects with the test succeeding.

If no test is specified, each object will be evaluated as a boolean.

Example usage:

.. sourcecode:: jinja

    {{ numbers|select("odd") }}
    {{ numbers|select("odd") }}
    {{ numbers|select("divisibleby", 3) }}
    {{ numbers|select("lessthan", 42) }}
    {{ strings|select("equalto", "mystring") }}

Similar to a generator comprehension such as:

.. code-block:: python

    (n for n in numbers if test_odd(n))
    (n for n in numbers if test_divisibleby(n, 3))

.. versionadded:: 2.7

### Function: sync_do_reject(context, value)

**Description:** Filters a sequence of objects by applying a test to each object,
and rejecting the objects with the test succeeding.

If no test is specified, each object will be evaluated as a boolean.

Example usage:

.. sourcecode:: jinja

    {{ numbers|reject("odd") }}

Similar to a generator comprehension such as:

.. code-block:: python

    (n for n in numbers if not test_odd(n))

.. versionadded:: 2.7

### Function: sync_do_selectattr(context, value)

**Description:** Filters a sequence of objects by applying a test to the specified
attribute of each object, and only selecting the objects with the
test succeeding.

If no test is specified, the attribute's value will be evaluated as
a boolean.

Example usage:

.. sourcecode:: jinja

    {{ users|selectattr("is_active") }}
    {{ users|selectattr("email", "none") }}

Similar to a generator comprehension such as:

.. code-block:: python

    (user for user in users if user.is_active)
    (user for user in users if test_none(user.email))

.. versionadded:: 2.7

### Function: sync_do_rejectattr(context, value)

**Description:** Filters a sequence of objects by applying a test to the specified
attribute of each object, and rejecting the objects with the test
succeeding.

If no test is specified, the attribute's value will be evaluated as
a boolean.

.. sourcecode:: jinja

    {{ users|rejectattr("is_active") }}
    {{ users|rejectattr("email", "none") }}

Similar to a generator comprehension such as:

.. code-block:: python

    (user for user in users if not user.is_active)
    (user for user in users if not test_none(user.email))

.. versionadded:: 2.7

### Function: do_tojson(eval_ctx, value, indent)

**Description:** Serialize an object to a string of JSON, and mark it safe to
render in HTML. This filter is only for use in HTML documents.

The returned string is safe to render in HTML documents and
``<script>`` tags. The exception is in HTML attributes that are
double quoted; either use single quotes or the ``|forceescape``
filter.

:param value: The object to serialize to JSON.
:param indent: The ``indent`` parameter passed to ``dumps``, for
    pretty-printing the value.

.. versionadded:: 2.9

### Function: prepare_map(context, args, kwargs)

### Function: prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)

### Function: select_or_reject(context, value, args, kwargs, modfunc, lookup_attr)

## Class: HasHTML

### Function: attrgetter(item)

### Function: attrgetter(item)

### Function: sort_func(item)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __html__(self)

### Function: func(x)

### Function: func(item)

### Function: transfunc(x)

### Function: func(item)
