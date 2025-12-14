## AI Summary

A file named test_pretty.py.


## Class: MyList

## Class: MyDict

## Class: MyObj

## Class: Dummy1

## Class: Dummy2

## Class: NoModule

## Class: Breaking

## Class: BreakingRepr

## Class: BadRepr

### Function: test_indentation()

**Description:** Test correct indentation in groups

### Function: test_dispatch()

**Description:** Test correct dispatching: The _repr_pretty_ method for MyDict
must be found before the registered printer for dict.

### Function: test_callability_checking()

**Description:** Test that the _repr_pretty_ method is tested for callability and skipped if
not.

### Function: test_sets(obj, expected_output)

**Description:** Test that set and frozenset use Python 3 formatting.

### Function: test_pprint_heap_allocated_type()

**Description:** Test that pprint works for heap allocated types.

### Function: test_pprint_nomod()

**Description:** Test that pprint works for classes with no __module__.

### Function: test_pprint_break()

**Description:** Test that p.break_ produces expected output

### Function: test_pprint_break_repr()

**Description:** Test that p.break_ is used in repr

### Function: test_bad_repr()

**Description:** Don't catch bad repr errors

## Class: BadException

## Class: ReallyBadRepr

### Function: test_really_bad_repr()

## Class: SA

## Class: SB

## Class: TestsPretty

## Class: MetaClass

### Function: test_metaclass_repr()

### Function: test_unicode_repr()

### Function: test_basic_class()

### Function: test_collections_userlist()

### Function: test_collections_defaultdict()

### Function: test_collections_ordereddict()

### Function: test_collections_deque()

### Function: test_collections_counter()

### Function: test_mappingproxy()

### Function: test_simplenamespace()

### Function: test_pretty_environ()

### Function: test_function_pretty()

**Description:** Test pretty print of function

## Class: OrderedCounter

**Description:** Counter that remembers the order elements are first encountered

## Class: MySet

### Function: test_custom_repr()

**Description:** A custom repr should override a pretty printer for a parent type

### Function: __init__(self, content)

### Function: _repr_pretty_(self, p, cycle)

### Function: _repr_pretty_(self, p, cycle)

### Function: somemethod(self)

### Function: _repr_pretty_(self, p, cycle)

### Function: _repr_pretty_(self, p, cycle)

### Function: __repr__(self)

### Function: __repr__(self)

### Function: __str__(self)

### Function: __class__(self)

### Function: __repr__(self)

### Function: test_super_repr(self)

### Function: test_long_list(self)

### Function: test_long_set(self)

### Function: test_long_tuple(self)

### Function: test_long_dict(self)

### Function: test_unbound_method(self)

### Function: __new__(cls, name)

### Function: __repr__(self)

## Class: C

### Function: type_pprint_wrapper(obj, p, cycle)

## Class: MyCounter

### Function: meaning_of_life(question)

### Function: __repr__(self)

### Function: __reduce__(self)

### Function: __repr__(self)

### Function: __repr__(self)
