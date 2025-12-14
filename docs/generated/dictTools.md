## AI Summary

A file named dictTools.py.


## Class: hashdict

**Description:** hashable dict implementation, suitable for use as a key into
other dicts.

    >>> h1 = hashdict({"apples": 1, "bananas":2})
    >>> h2 = hashdict({"bananas": 3, "mangoes": 5})
    >>> h1+h2
    hashdict(apples=1, bananas=3, mangoes=5)
    >>> d1 = {}
    >>> d1[h1] = "salad"
    >>> d1[h1]
    'salad'
    >>> d1[h2]
    Traceback (most recent call last):
    ...
    KeyError: hashdict(bananas=3, mangoes=5)

based on answers from
   http://stackoverflow.com/questions/1151658/python-hashable-dicts

### Function: __key(self)

### Function: __repr__(self)

### Function: __hash__(self)

### Function: __setitem__(self, key, value)

### Function: __delitem__(self, key)

### Function: clear(self)

### Function: pop(self)

### Function: popitem(self)

### Function: setdefault(self)

### Function: update(self)

### Function: __add__(self, right)
