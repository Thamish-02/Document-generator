## AI Summary

A file named table_builder.py.


## Class: BuildCallback

**Description:** Keyed on (BEFORE_BUILD, class[, Format if available]).
Receives (dest, source).
Should return (dest, source), which can be new objects.

### Function: _assignable(convertersByName)

### Function: _isNonStrSequence(value)

### Function: _split_format(cls, source)

## Class: TableBuilder

**Description:** Helps to populate things derived from BaseTable from maps, tuples, etc.

A table of lifecycle callbacks may be provided to add logic beyond what is possible
based on otData info for the target class. See BuildCallbacks.

## Class: TableUnbuilder

### Function: __init__(self, callbackTable)

### Function: _convert(self, dest, field, converter, value)

### Function: build(self, cls, source)

### Function: __init__(self, callbackTable)

### Function: unbuild(self, table)
