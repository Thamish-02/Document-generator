## AI Summary

A file named statNames.py.


## Class: StatNames

**Description:** Name data generated from the STAT table information.

### Function: getStatNames(doc, userLocation)

**Description:** Compute the family, style, PostScript names of the given ``userLocation``
using the document's STAT information.

Also computes localizations.

If not enough STAT data is available for a given name, either its dict of
localized names will be empty (family and style names), or the name will be
None (PostScript name).

Note: this method does not consider info attached to the instance, like
family name. The user needs to override all names on an instance that STAT
information would compute differently than desired.

.. versionadded:: 5.0

### Function: _getSortedAxisLabels(axes)

**Description:** Returns axis labels sorted by their ordering, with unordered ones appended as
they are listed.

### Function: _getAxisLabelsForUserLocation(axes, userLocation)

### Function: _getRibbiStyle(self, userLocation)

**Description:** Compute the RIBBI style name of the given user location,
return the location of the matching Regular in the RIBBI group.

.. versionadded:: 5.0
