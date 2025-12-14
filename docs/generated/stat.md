## AI Summary

A file named stat.py.


### Function: buildVFStatTable(ttFont, doc, vfName)

**Description:** Build the STAT table for the variable font identified by its name in
the given document.

Knowing which variable we're building STAT data for is needed to subset
the STAT locations to only include what the variable font actually ships.

.. versionadded:: 5.0

.. seealso::
    - :func:`getStatAxes()`
    - :func:`getStatLocations()`
    - :func:`fontTools.otlLib.builder.buildStatTable()`

### Function: getStatAxes(doc, userRegion)

**Description:** Return a list of axis dicts suitable for use as the ``axes``
argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

.. versionadded:: 5.0

### Function: getStatLocations(doc, userRegion)

**Description:** Return a list of location dicts suitable for use as the ``locations``
argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

.. versionadded:: 5.0

### Function: _labelToFlags(label)

### Function: _axisLabelToStatLocation(label)
