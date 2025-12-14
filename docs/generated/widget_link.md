## AI Summary

A file named widget_link.py.


## Class: WidgetTraitTuple

**Description:** Traitlet for validating a single (Widget, 'trait_name') pair

## Class: Link

**Description:** Link Widget

source: a (Widget, 'trait_name') tuple for the source trait
target: a (Widget, 'trait_name') tuple that should be updated

### Function: jslink(attr1, attr2)

**Description:** Link two widget attributes on the frontend so they remain in sync.

The link is created in the front-end and does not rely on a roundtrip
to the backend.

Parameters
----------
source : a (Widget, 'trait_name') tuple for the first trait
target : a (Widget, 'trait_name') tuple for the second trait

Examples
--------

>>> c = link((widget1, 'value'), (widget2, 'value'))

## Class: DirectionalLink

**Description:** A directional link

source: a (Widget, 'trait_name') tuple for the source trait
target: a (Widget, 'trait_name') tuple that should be updated
when the source trait changes.

### Function: jsdlink(source, target)

**Description:** Link a source widget attribute with a target widget attribute.

The link is created in the front-end and does not rely on a roundtrip
to the backend.

Parameters
----------
source : a (Widget, 'trait_name') tuple for the source trait
target : a (Widget, 'trait_name') tuple for the target trait

Examples
--------

>>> c = dlink((src_widget, 'value'), (tgt_widget, 'value'))

### Function: __init__(self)

### Function: validate_elements(self, obj, value)

### Function: __init__(self, source, target)

### Function: unlink(self)
