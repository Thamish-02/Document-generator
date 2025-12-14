## AI Summary

A file named cm.py.


### Function: _gen_cmap_registry()

**Description:** Generate a dict mapping standard colormap names to standard colormaps, as
well as the reversed colormaps.

## Class: ColormapRegistry

**Description:** Container for colormaps that are known to Matplotlib by name.

The universal registry instance is `matplotlib.colormaps`. There should be
no need for users to instantiate `.ColormapRegistry` themselves.

Read access uses a dict-like interface mapping names to `.Colormap`\s::

    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']

Returned `.Colormap`\s are copies, so that their modification does not
change the global definition of the colormap.

Additional colormaps can be added via `.ColormapRegistry.register`::

    mpl.colormaps.register(my_colormap)

To get a list of all registered colormaps, you can do::

    from matplotlib import colormaps
    list(colormaps)

### Function: get_cmap(name, lut)

**Description:** Get a colormap instance, defaulting to rc values if *name* is None.

Parameters
----------
name : `~matplotlib.colors.Colormap` or str or None, default: None
    If a `.Colormap` instance, it will be returned. Otherwise, the name of
    a colormap known to Matplotlib, which will be resampled by *lut*. The
    default, None, means :rc:`image.cmap`.
lut : int or None, default: None
    If *name* is not already a Colormap instance and *lut* is not None, the
    colormap will be resampled to have *lut* entries in the lookup table.

Returns
-------
Colormap

### Function: _ensure_cmap(cmap)

**Description:** Ensure that we have a `.Colormap` object.

For internal use to preserve type stability of errors.

Parameters
----------
cmap : None, str, Colormap

    - if a `Colormap`, return it
    - if a string, look it up in mpl.colormaps
    - if None, look up the default color map in mpl.colormaps

Returns
-------
Colormap

### Function: __init__(self, cmaps)

### Function: __getitem__(self, item)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __str__(self)

### Function: __call__(self)

**Description:** Return a list of the registered colormap names.

This exists only for backward-compatibility in `.pyplot` which had a
``plt.colormaps()`` method. The recommended way to get this list is
now ``list(colormaps)``.

### Function: register(self, cmap)

**Description:** Register a new colormap.

The colormap name can then be used as a string argument to any ``cmap``
parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

The colormap registry stores a copy of the given colormap, so that
future changes to the original colormap instance do not affect the
registered colormap. Think of this as the registry taking a snapshot
of the colormap at registration.

Parameters
----------
cmap : matplotlib.colors.Colormap
    The colormap to register.

name : str, optional
    The name for the colormap. If not given, ``cmap.name`` is used.

force : bool, default: False
    If False, a ValueError is raised if trying to overwrite an already
    registered name. True supports overwriting registered colormaps
    other than the builtin colormaps.

### Function: unregister(self, name)

**Description:** Remove a colormap from the registry.

You cannot remove built-in colormaps.

If the named colormap is not registered, returns with no error, raises
if you try to de-register a default colormap.

.. warning::

    Colormap names are currently a shared namespace that may be used
    by multiple packages. Use `unregister` only if you know you
    have registered that name before. In particular, do not
    unregister just in case to clean the name before registering a
    new colormap.

Parameters
----------
name : str
    The name of the colormap to be removed.

Raises
------
ValueError
    If you try to remove a default built-in colormap.

### Function: get_cmap(self, cmap)

**Description:** Return a color map specified through *cmap*.

Parameters
----------
cmap : str or `~matplotlib.colors.Colormap` or None

    - if a `.Colormap`, return it
    - if a string, look it up in ``mpl.colormaps``
    - if None, return the Colormap defined in :rc:`image.cmap`

Returns
-------
Colormap
