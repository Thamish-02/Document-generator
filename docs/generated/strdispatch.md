## AI Summary

A file named strdispatch.py.


## Class: StrDispatch

**Description:** Dispatch (lookup) a set of strings / regexps for match.

Example:

>>> dis = StrDispatch()
>>> dis.add_s('hei',34, priority = 4)
>>> dis.add_s('hei',123, priority = 2)
>>> dis.add_re('h.i', 686)
>>> print(list(dis.flat_matches('hei')))
[123, 34, 686]

### Function: __init__(self)

### Function: add_s(self, s, obj, priority)

**Description:** Adds a target 'string' for dispatching 

### Function: add_re(self, regex, obj, priority)

**Description:** Adds a target regexp for dispatching 

### Function: dispatch(self, key)

**Description:** Get a seq of Commandchain objects that match key 

### Function: __repr__(self)

### Function: s_matches(self, key)

### Function: flat_matches(self, key)

**Description:** Yield all 'value' targets, without priority 
