## AI Summary

A file named split.py.


### Function: defaultMakeInstanceFilename(doc, instance, statNames)

**Description:** Default callable to synthesize an instance filename
when makeNames=True, for instances that don't specify an instance name
in the designspace. This part of the name generation can be overriden
because it's not specified by the STAT table.

### Function: splitInterpolable(doc, makeNames, expandLocations, makeInstanceFilename)

**Description:** Split the given DS5 into several interpolable sub-designspaces.
There are as many interpolable sub-spaces as there are combinations of
discrete axis values.

E.g. with axes:
    - italic (discrete) Upright or Italic
    - style (discrete) Sans or Serif
    - weight (continuous) 100 to 900

There are 4 sub-spaces in which the Weight axis should interpolate:
(Upright, Sans), (Upright, Serif), (Italic, Sans) and (Italic, Serif).

The sub-designspaces still include the full axis definitions and STAT data,
but the rules, sources, variable fonts, instances are trimmed down to only
keep what falls within the interpolable sub-space.

Args:
  - ``makeNames``: Whether to compute the instance family and style
    names using the STAT data.
  - ``expandLocations``: Whether to turn all locations into "full"
    locations, including implicit default axis values where missing.
  - ``makeInstanceFilename``: Callable to synthesize an instance filename
    when makeNames=True, for instances that don't specify an instance name
    in the designspace. This part of the name generation can be overridden
    because it's not specified by the STAT table.

.. versionadded:: 5.0

### Function: splitVariableFonts(doc, makeNames, expandLocations, makeInstanceFilename)

**Description:** Convert each variable font listed in this document into a standalone
designspace. This can be used to compile all the variable fonts from a
format 5 designspace using tools that can only deal with 1 VF at a time.

Args:
  - ``makeNames``: Whether to compute the instance family and style
    names using the STAT data.
  - ``expandLocations``: Whether to turn all locations into "full"
    locations, including implicit default axis values where missing.
  - ``makeInstanceFilename``: Callable to synthesize an instance filename
    when makeNames=True, for instances that don't specify an instance name
    in the designspace. This part of the name generation can be overridden
    because it's not specified by the STAT table.

.. versionadded:: 5.0

### Function: convert5to4(doc)

**Description:** Convert each variable font listed in this document into a standalone
format 4 designspace. This can be used to compile all the variable fonts
from a format 5 designspace using tools that only know about format 4.

.. versionadded:: 5.0

### Function: _extractSubSpace(doc, userRegion)

### Function: _conditionSetFrom(conditionSet)

### Function: _subsetRulesBasedOnConditions(rules, designRegion)

### Function: _filterLocation(userRegion, location)

### Function: maybeExpandDesignLocation(object)
