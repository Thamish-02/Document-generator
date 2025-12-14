## AI Summary

A file named _pickle.py.


### Function: __bit_generator_ctor(bit_generator)

**Description:** Pickling helper function that returns a bit generator object

Parameters
----------
bit_generator : type[BitGenerator] or str
    BitGenerator class or string containing the name of the BitGenerator

Returns
-------
BitGenerator
    BitGenerator instance

### Function: __generator_ctor(bit_generator_name, bit_generator_ctor)

**Description:** Pickling helper function that returns a Generator object

Parameters
----------
bit_generator_name : str or BitGenerator
    String containing the core BitGenerator's name or a
    BitGenerator instance
bit_generator_ctor : callable, optional
    Callable function that takes bit_generator_name as its only argument
    and returns an instantized bit generator.

Returns
-------
rg : Generator
    Generator using the named core BitGenerator

### Function: __randomstate_ctor(bit_generator_name, bit_generator_ctor)

**Description:** Pickling helper function that returns a legacy RandomState-like object

Parameters
----------
bit_generator_name : str
    String containing the core BitGenerator's name
bit_generator_ctor : callable, optional
    Callable function that takes bit_generator_name as its only argument
    and returns an instantized bit generator.

Returns
-------
rs : RandomState
    Legacy RandomState using the named core BitGenerator
