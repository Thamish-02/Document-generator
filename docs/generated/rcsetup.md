## AI Summary

A file named rcsetup.py.


## Class: __getattr__

## Class: ValidateInStrings

### Function: _listify_validator(scalar_validator, allow_stringlist)

### Function: validate_any(s)

### Function: _validate_date(s)

### Function: validate_bool(b)

**Description:** Convert b to ``bool`` or raise.

### Function: validate_axisbelow(s)

### Function: validate_dpi(s)

**Description:** Confirm s is string 'figure' or convert s to float or raise.

### Function: _make_type_validator(cls)

**Description:** Return a validator that converts inputs to *cls* or raises (and possibly
allows ``None`` as well).

### Function: _validate_marker(s)

### Function: _validate_pathlike(s)

### Function: validate_fonttype(s)

**Description:** Confirm that this is a Postscript or PDF font type that we know how to
convert to.

### Function: validate_backend(s)

### Function: _validate_toolbar(s)

### Function: validate_color_or_inherit(s)

**Description:** Return a valid color arg.

### Function: validate_color_or_auto(s)

### Function: validate_color_for_prop_cycle(s)

### Function: _validate_color_or_linecolor(s)

### Function: validate_color(s)

**Description:** Return a valid color arg.

### Function: _validate_cmap(s)

### Function: validate_aspect(s)

### Function: validate_fontsize_None(s)

### Function: validate_fontsize(s)

### Function: validate_fontweight(s)

### Function: validate_fontstretch(s)

### Function: validate_font_properties(s)

### Function: _validate_mathtext_fallback(s)

### Function: validate_whiskers(s)

### Function: validate_ps_distiller(s)

### Function: _validate_linestyle(ls)

**Description:** A validator for all possible line styles, the named ones *and*
the on-off ink sequences.

### Function: validate_markevery(s)

**Description:** Validate the markevery property of a Line2D object.

Parameters
----------
s : None, int, (int, int), slice, float, (float, float), or list[int]

Returns
-------
None, int, (int, int), slice, float, (float, float), or list[int]

### Function: validate_bbox(s)

### Function: validate_sketch(s)

### Function: _validate_greaterthan_minushalf(s)

### Function: _validate_greaterequal0_lessequal1(s)

### Function: _validate_int_greaterequal0(s)

### Function: validate_hatch(s)

**Description:** Validate a hatch pattern.
A hatch pattern string can have any sequence of the following
characters: ``\ / | - + * . x o O``.

### Function: _validate_minor_tick_ndivs(n)

**Description:** Validate ndiv parameter related to the minor ticks.
It controls the number of minor ticks to be placed between
two major ticks.

### Function: cycler()

**Description:** Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
but includes input validation.

Call signatures::

  cycler(cycler)
  cycler(label=values, label2=values2, ...)
  cycler(label, values)

Form 1 copies a given `~cycler.Cycler` object.

Form 2 creates a `~cycler.Cycler` which cycles over one or more
properties simultaneously. If multiple properties are given, their
value lists must have the same length.

Form 3 creates a `~cycler.Cycler` for a single property. This form
exists for compatibility with the original cycler. Its use is
discouraged in favor of the kwarg form, i.e. ``cycler(label=values)``.

Parameters
----------
cycler : Cycler
    Copy constructor for Cycler.

label : str
    The property key. Must be a valid `.Artist` property.
    For example, 'color' or 'linestyle'. Aliases are allowed,
    such as 'c' for 'color' and 'lw' for 'linewidth'.

values : iterable
    Finite-length iterable of the property values. These values
    are validated and will raise a ValueError if invalid.

Returns
-------
Cycler
    A new :class:`~cycler.Cycler` for the given properties.

Examples
--------
Creating a cycler for a single property:

>>> c = cycler(color=['red', 'green', 'blue'])

Creating a cycler for simultaneously cycling over multiple properties
(e.g. red circle, green plus, blue cross):

>>> c = cycler(color=['red', 'green', 'blue'],
...            marker=['o', '+', 'x'])

## Class: _DunderChecker

### Function: _validate_legend_loc(loc)

**Description:** Confirm that loc is a type which rc.Params["legend.loc"] supports.

.. versionadded:: 3.8

Parameters
----------
loc : str | int | (float, float) | str((float, float))
    The location of the legend.

Returns
-------
loc : str | int | (float, float) or raise ValueError exception
    The location of the legend.

### Function: validate_cycler(s)

**Description:** Return a Cycler object from a string repr or the object itself.

### Function: validate_hist_bins(s)

## Class: _ignorecase

**Description:** A marker class indicating that a list-of-str is case-insensitive.

### Function: _convert_validator_spec(key, conv)

### Function: interactive_bk(self)

### Function: non_interactive_bk(self)

### Function: all_backends(self)

### Function: __init__(self, key, valid, ignorecase)

**Description:** *valid* is a list of legal strings.

### Function: __call__(self, s)

### Function: f(s)

### Function: validator(s)

### Function: _is_iterable_not_string_like(x)

### Function: visit_Attribute(self, node)

### Function: func(s)
