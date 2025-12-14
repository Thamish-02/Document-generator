## AI Summary

A file named six.py.


### Function: _add_doc(func, doc)

**Description:** Add documentation to a function.

### Function: _import_module(name)

**Description:** Import module, returning the module after the last dot.

## Class: _LazyDescr

## Class: MovedModule

## Class: _LazyModule

## Class: MovedAttribute

## Class: _SixMetaPathImporter

**Description:** A meta path importer to import six.moves and its submodules.

This class implements a PEP302 finder and loader. It should be compatible
with Python 2.5 and all existing versions of Python3

## Class: _MovedItems

**Description:** Lazy loading of moved objects

## Class: Module_six_moves_urllib_parse

**Description:** Lazy loading of moved objects in six.moves.urllib_parse

## Class: Module_six_moves_urllib_error

**Description:** Lazy loading of moved objects in six.moves.urllib_error

## Class: Module_six_moves_urllib_request

**Description:** Lazy loading of moved objects in six.moves.urllib_request

## Class: Module_six_moves_urllib_response

**Description:** Lazy loading of moved objects in six.moves.urllib_response

## Class: Module_six_moves_urllib_robotparser

**Description:** Lazy loading of moved objects in six.moves.urllib_robotparser

## Class: Module_six_moves_urllib

**Description:** Create a six.moves.urllib namespace that resembles the Python 3 namespace

### Function: add_move(move)

**Description:** Add an item to six.moves.

### Function: remove_move(name)

**Description:** Remove item from six.moves.

### Function: assertCountEqual(self)

### Function: assertRaisesRegex(self)

### Function: assertRegex(self)

### Function: assertNotRegex(self)

### Function: with_metaclass(meta)

**Description:** Create a base class with a metaclass.

### Function: add_metaclass(metaclass)

**Description:** Class decorator for creating a class with a metaclass.

### Function: ensure_binary(s, encoding, errors)

**Description:** Coerce **s** to six.binary_type.

For Python 2:
  - `unicode` -> encoded to `str`
  - `str` -> `str`

For Python 3:
  - `str` -> encoded to `bytes`
  - `bytes` -> `bytes`

### Function: ensure_str(s, encoding, errors)

**Description:** Coerce *s* to `str`.

For Python 2:
  - `unicode` -> encoded to `str`
  - `str` -> `str`

For Python 3:
  - `str` -> `str`
  - `bytes` -> decoded to `str`

### Function: ensure_text(s, encoding, errors)

**Description:** Coerce *s* to six.text_type.

For Python 2:
  - `unicode` -> `unicode`
  - `str` -> `unicode`

For Python 3:
  - `str` -> `str`
  - `bytes` -> decoded to `str`

### Function: python_2_unicode_compatible(klass)

**Description:** A class decorator that defines __unicode__ and __str__ methods under Python 2.
Under Python 3 it does nothing.

To support Python 2 and 3 with a single code base, define a __str__ method
returning text and apply this decorator to the class.

### Function: __init__(self, name)

### Function: __get__(self, obj, tp)

### Function: __init__(self, name, old, new)

### Function: _resolve(self)

### Function: __getattr__(self, attr)

### Function: __init__(self, name)

### Function: __dir__(self)

### Function: __init__(self, name, old_mod, new_mod, old_attr, new_attr)

### Function: _resolve(self)

### Function: __init__(self, six_module_name)

### Function: _add_module(self, mod)

### Function: _get_module(self, fullname)

### Function: find_module(self, fullname, path)

### Function: find_spec(self, fullname, path, target)

### Function: __get_module(self, fullname)

### Function: load_module(self, fullname)

### Function: is_package(self, fullname)

**Description:** Return true, if the named module is a package.

We need this method to get correct spec objects with
Python 3.4 (see PEP451)

### Function: get_code(self, fullname)

**Description:** Return None

Required, if is_package is implemented

### Function: create_module(self, spec)

### Function: exec_module(self, module)

### Function: __dir__(self)

### Function: get_unbound_function(unbound)

### Function: create_unbound_method(func, cls)

### Function: get_unbound_function(unbound)

### Function: create_bound_method(func, obj)

### Function: create_unbound_method(func, cls)

## Class: Iterator

### Function: iterkeys(d)

### Function: itervalues(d)

### Function: iteritems(d)

### Function: iterlists(d)

### Function: iterkeys(d)

### Function: itervalues(d)

### Function: iteritems(d)

### Function: iterlists(d)

### Function: b(s)

### Function: u(s)

### Function: b(s)

### Function: u(s)

### Function: byte2int(bs)

### Function: indexbytes(buf, i)

### Function: reraise(tp, value, tb)

### Function: exec_(_code_, _globs_, _locs_)

**Description:** Execute code in a namespace.

### Function: raise_from(value, from_value)

### Function: print_()

**Description:** The new-style print function for Python 2.4 and 2.5.

### Function: print_()

### Function: _update_wrapper(wrapper, wrapped, assigned, updated)

### Function: wraps(wrapped, assigned, updated)

## Class: metaclass

### Function: wrapper(cls)

## Class: X

### Function: advance_iterator(it)

### Function: callable(obj)

### Function: next(self)

### Function: write(data)

### Function: __new__(cls, name, this_bases, d)

### Function: __prepare__(cls, name, this_bases)

### Function: __len__(self)
