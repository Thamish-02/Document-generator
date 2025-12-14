## AI Summary

A file named jsonpointer.py.


### Function: set_pointer(doc, pointer, value, inplace)

**Description:** Resolves a pointer against doc and sets the value of the target within doc.

With inplace set to true, doc is modified as long as pointer is not the
root.

>>> obj = {'foo': {'anArray': [ {'prop': 44}], 'another prop': {'baz': 'A string' }}}

>>> set_pointer(obj, '/foo/anArray/0/prop', 55) ==     {'foo': {'another prop': {'baz': 'A string'}, 'anArray': [{'prop': 55}]}}
True

>>> set_pointer(obj, '/foo/yet another prop', 'added prop') ==     {'foo': {'another prop': {'baz': 'A string'}, 'yet another prop': 'added prop', 'anArray': [{'prop': 55}]}}
True

>>> obj = {'foo': {}}
>>> set_pointer(obj, '/foo/a%20b', 'x') ==     {'foo': {'a%20b': 'x' }}
True

### Function: resolve_pointer(doc, pointer, default)

**Description:** Resolves pointer against doc and returns the referenced object

>>> obj = {'foo': {'anArray': [ {'prop': 44}], 'another prop': {'baz': 'A string' }}, 'a%20b': 1, 'c d': 2}

>>> resolve_pointer(obj, '') == obj
True

>>> resolve_pointer(obj, '/foo') == obj['foo']
True

>>> resolve_pointer(obj, '/foo/another prop') == obj['foo']['another prop']
True

>>> resolve_pointer(obj, '/foo/another prop/baz') == obj['foo']['another prop']['baz']
True

>>> resolve_pointer(obj, '/foo/anArray/0') == obj['foo']['anArray'][0]
True

>>> resolve_pointer(obj, '/some/path', None) == None
True

>>> resolve_pointer(obj, '/a b', None) == None
True

>>> resolve_pointer(obj, '/a%20b') == 1
True

>>> resolve_pointer(obj, '/c d') == 2
True

>>> resolve_pointer(obj, '/c%20d', None) == None
True

### Function: pairwise(iterable)

**Description:** Transforms a list to a list of tuples of adjacent items

s -> (s0,s1), (s1,s2), (s2, s3), ...

>>> list(pairwise([]))
[]

>>> list(pairwise([1]))
[]

>>> list(pairwise([1, 2, 3, 4]))
[(1, 2), (2, 3), (3, 4)]

## Class: JsonPointerException

## Class: EndOfList

**Description:** Result of accessing element "-" of a list

## Class: JsonPointer

**Description:** A JSON Pointer that can reference parts of a JSON document

### Function: escape(s)

### Function: unescape(s)

### Function: __init__(self, list_)

### Function: __repr__(self)

### Function: __init__(self, pointer)

### Function: to_last(self, doc)

**Description:** Resolves ptr until the last step, returns (sub-doc, last-step)

### Function: resolve(self, doc, default)

**Description:** Resolves the pointer against doc and returns the referenced object

### Function: set(self, doc, value, inplace)

**Description:** Resolve the pointer against the doc and replace the target with value.

### Function: get_part(cls, doc, part)

**Description:** Returns the next step in the correct type

### Function: get_parts(self)

**Description:** Returns the list of the parts. For example, JsonPointer('/a/b').get_parts() == ['a', 'b']

### Function: walk(self, doc, part)

**Description:** Walks one step in doc and returns the referenced part 

### Function: contains(self, ptr)

**Description:** Returns True if self contains the given ptr 

### Function: __contains__(self, item)

**Description:** Returns True if self contains the given ptr 

### Function: join(self, suffix)

**Description:** Returns a new JsonPointer with the given suffix append to this ptr 

### Function: __truediv__(self, suffix)

### Function: path(self)

**Description:** Returns the string representation of the pointer

>>> ptr = JsonPointer('/~0/0/~1').path == '/~0/0/~1'

### Function: __eq__(self, other)

**Description:** Compares a pointer to another object

Pointers can be compared by comparing their strings (or splitted
strings), because no two different parts can point to the same
structure in an object (eg no different number representations)

### Function: __hash__(self)

### Function: __str__(self)

### Function: __repr__(self)

### Function: from_parts(cls, parts)

**Description:** Constructs a JsonPointer from a list of (unescaped) paths

>>> JsonPointer.from_parts(['a', '~', '/', 0]).path == '/a/~0/~1/0'
True
