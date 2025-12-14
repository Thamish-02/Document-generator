## AI Summary

A file named configTools.py.


## Class: ConfigError

**Description:** Base exception for the config module.

## Class: ConfigAlreadyRegisteredError

**Description:** Raised when a module tries to register a configuration option that
already exists.

Should not be raised too much really, only when developing new fontTools
modules.

## Class: ConfigValueParsingError

**Description:** Raised when a configuration value cannot be parsed.

## Class: ConfigValueValidationError

**Description:** Raised when a configuration value cannot be validated.

## Class: ConfigUnknownOptionError

**Description:** Raised when a configuration option is unknown.

## Class: Option

## Class: Options

**Description:** Registry of available options for a given config system.

Define new options using the :meth:`register()` method.

Access existing options using the Mapping interface.

## Class: AbstractConfig

**Description:** Create a set of config values, optionally pre-filled with values from
the given dictionary or pre-existing config object.

The class implements the MutableMapping protocol keyed by option name (`str`).
For convenience its methods accept either Option or str as the key parameter.

.. seealso:: :meth:`set()`

This config class is abstract because it needs its ``options`` class
var to be set to an instance of :class:`Options` before it can be
instanciated and used.

.. code:: python

    class MyConfig(AbstractConfig):
        options = Options()

    MyConfig.register_option( "test:option_name", "This is an option", 0, int, lambda v: isinstance(v, int))

    cfg = MyConfig({"test:option_name": 10})

### Function: __init__(self, name)

### Function: __init__(self, name, value)

### Function: __init__(self, name, value)

### Function: __init__(self, option_or_name)

### Function: parse_optional_bool(v)

### Function: validate_optional_bool(v)

### Function: __init__(self, other)

### Function: register(self, name, help, default, parse, validate)

**Description:** Create and register a new option.

### Function: register_option(self, option)

**Description:** Register a new option.

### Function: is_registered(self, option)

**Description:** Return True if the same option object is already registered.

### Function: __getitem__(self, key)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __repr__(self)

### Function: register_option(cls, name, help, default, parse, validate)

**Description:** Register an available option in this config system.

### Function: __init__(self, values, parse_values, skip_unknown)

### Function: _resolve_option(self, option_or_name)

### Function: set(self, option_or_name, value, parse_values, skip_unknown)

**Description:** Set the value of an option.

Args:
    * `option_or_name`: an `Option` object or its name (`str`).
    * `value`: the value to be assigned to given option.
    * `parse_values`: parse the configuration value from a string into
        its proper type, as per its `Option` object. The default
        behavior is to raise `ConfigValueValidationError` when the value
        is not of the right type. Useful when reading options from a
        file type that doesn't support as many types as Python.
    * `skip_unknown`: skip unknown configuration options. The default
        behaviour is to raise `ConfigUnknownOptionError`. Useful when
        reading options from a configuration file that has extra entries
        (e.g. for a later version of fontTools)

### Function: get(self, option_or_name, default)

**Description:** Get the value of an option. The value which is returned is the first
provided among:

1. a user-provided value in the options's ``self._values`` dict
2. a caller-provided default value to this method call
3. the global default for the option provided in ``fontTools.config``

This is to provide the ability to migrate progressively from config
options passed as arguments to fontTools APIs to config options read
from the current TTFont, e.g.

.. code:: python

    def fontToolsAPI(font, some_option):
        value = font.cfg.get("someLib.module:SOME_OPTION", some_option)
        # use value

That way, the function will work the same for users of the API that
still pass the option to the function call, but will favour the new
config mechanism if the given font specifies a value for that option.

### Function: copy(self)

### Function: __getitem__(self, option_or_name)

### Function: __setitem__(self, option_or_name, value)

### Function: __delitem__(self, option_or_name)

### Function: __iter__(self)

### Function: __len__(self)

### Function: __repr__(self)
