## AI Summary

A file named classifyTools.py.


## Class: Classifier

**Description:** Main Classifier object, used to classify things into similar sets.

### Function: classify(list_of_sets, sort)

**Description:** Takes a iterable of iterables (list of sets from here on; but any
iterable works.), and returns the smallest list of sets such that
each set, is either a subset, or is disjoint from, each of the input
sets.

In other words, this function classifies all the things present in
any of the input sets, into similar classes, based on which sets
things are a member of.

If sort=True, return class sets are sorted by decreasing size and
their natural sort order within each class size.  Otherwise, class
sets are returned in the order that they were identified, which is
generally not significant.

>>> classify([]) == ([], {})
True
>>> classify([[]]) == ([], {})
True
>>> classify([[], []]) == ([], {})
True
>>> classify([[1]]) == ([{1}], {1: {1}})
True
>>> classify([[1,2]]) == ([{1, 2}], {1: {1, 2}, 2: {1, 2}})
True
>>> classify([[1],[2]]) == ([{1}, {2}], {1: {1}, 2: {2}})
True
>>> classify([[1,2],[2]]) == ([{1}, {2}], {1: {1}, 2: {2}})
True
>>> classify([[1,2],[2,4]]) == ([{1}, {2}, {4}], {1: {1}, 2: {2}, 4: {4}})
True
>>> classify([[1,2],[2,4,5]]) == (
...     [{4, 5}, {1}, {2}], {1: {1}, 2: {2}, 4: {4, 5}, 5: {4, 5}})
True
>>> classify([[1,2],[2,4,5]], sort=False) == (
...     [{1}, {4, 5}, {2}], {1: {1}, 2: {2}, 4: {4, 5}, 5: {4, 5}})
True
>>> classify([[1,2,9],[2,4,5]], sort=False) == (
...     [{1, 9}, {4, 5}, {2}], {1: {1, 9}, 2: {2}, 4: {4, 5}, 5: {4, 5},
...     9: {1, 9}})
True
>>> classify([[1,2,9,15],[2,4,5]], sort=False) == (
...     [{1, 9, 15}, {4, 5}, {2}], {1: {1, 9, 15}, 2: {2}, 4: {4, 5},
...     5: {4, 5}, 9: {1, 9, 15}, 15: {1, 9, 15}})
True
>>> classes, mapping = classify([[1,2,9,15],[2,4,5],[15,5]], sort=False)
>>> set([frozenset(c) for c in classes]) == set(
...     [frozenset(s) for s in ({1, 9}, {4}, {2}, {5}, {15})])
True
>>> mapping == {1: {1, 9}, 2: {2}, 4: {4}, 5: {5}, 9: {1, 9}, 15: {15}}
True

### Function: __init__(self, sort)

### Function: add(self, set_of_things)

**Description:** Add a set to the classifier.  Any iterable is accepted.

### Function: update(self, list_of_sets)

**Description:** Add a a list of sets to the classifier.  Any iterable of iterables is accepted.

### Function: _process(self)

### Function: getThings(self)

**Description:** Returns the set of all things known so far.

The return value belongs to the Classifier object and should NOT
be modified while the classifier is still in use.

### Function: getMapping(self)

**Description:** Returns the mapping from things to their class set.

The return value belongs to the Classifier object and should NOT
be modified while the classifier is still in use.

### Function: getClasses(self)

**Description:** Returns the list of class sets.

The return value belongs to the Classifier object and should NOT
be modified while the classifier is still in use.
