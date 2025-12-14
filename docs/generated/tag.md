## AI Summary

A file named tag.py.


## Class: JSONTag

**Description:** Base class for defining type tags for :class:`TaggedJSONSerializer`.

## Class: TagDict

**Description:** Tag for 1-item dicts whose only key matches a registered tag.

Internally, the dict key is suffixed with `__`, and the suffix is removed
when deserializing.

## Class: PassDict

## Class: TagTuple

## Class: PassList

## Class: TagBytes

## Class: TagMarkup

**Description:** Serialize anything matching the :class:`~markupsafe.Markup` API by
having a ``__html__`` method to the result of that method. Always
deserializes to an instance of :class:`~markupsafe.Markup`.

## Class: TagUUID

## Class: TagDateTime

## Class: TaggedJSONSerializer

**Description:** Serializer that uses a tag system to compactly represent objects that
are not JSON types. Passed as the intermediate serializer to
:class:`itsdangerous.Serializer`.

The following extra types are supported:

* :class:`dict`
* :class:`tuple`
* :class:`bytes`
* :class:`~markupsafe.Markup`
* :class:`~uuid.UUID`
* :class:`~datetime.datetime`

### Function: __init__(self, serializer)

**Description:** Create a tagger for the given serializer.

### Function: check(self, value)

**Description:** Check if the given value should be tagged by this tag.

### Function: to_json(self, value)

**Description:** Convert the Python object to an object that is a valid JSON type.
The tag will be added later.

### Function: to_python(self, value)

**Description:** Convert the JSON representation back to the correct type. The tag
will already be removed.

### Function: tag(self, value)

**Description:** Convert the value to a valid JSON type and add the tag structure
around it.

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: check(self, value)

### Function: to_json(self, value)

### Function: to_python(self, value)

### Function: __init__(self)

### Function: register(self, tag_class, force, index)

**Description:** Register a new tag with this serializer.

:param tag_class: tag class to register. Will be instantiated with this
    serializer instance.
:param force: overwrite an existing tag. If false (default), a
    :exc:`KeyError` is raised.
:param index: index to insert the new tag in the tag order. Useful when
    the new tag is a special case of an existing tag. If ``None``
    (default), the tag is appended to the end of the order.

:raise KeyError: if the tag key is already registered and ``force`` is
    not true.

### Function: tag(self, value)

**Description:** Convert a value to a tagged representation if necessary.

### Function: untag(self, value)

**Description:** Convert a tagged representation back to the original type.

### Function: _untag_scan(self, value)

### Function: dumps(self, value)

**Description:** Tag the value and dump it to a compact JSON string.

### Function: loads(self, value)

**Description:** Load data from a JSON string and deserialized any tagged objects.
