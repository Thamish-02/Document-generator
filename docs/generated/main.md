## AI Summary

A file named main.py.


### Function: _load_dotenv_disabled()

**Description:** Determine if dotenv loading has been disabled.

### Function: with_warn_for_invalid_lines(mappings)

## Class: DotEnv

### Function: get_key(dotenv_path, key_to_get, encoding)

**Description:** Get the value of a given key from the given .env.

Returns `None` if the key isn't found or doesn't have a value.

### Function: rewrite(path, encoding)

### Function: set_key(dotenv_path, key_to_set, value_to_set, quote_mode, export, encoding)

**Description:** Adds or Updates a key/value to the given .env

If the .env path given doesn't exist, fails instead of risking creating
an orphan .env somewhere in the filesystem

### Function: unset_key(dotenv_path, key_to_unset, quote_mode, encoding)

**Description:** Removes a given key from the given `.env` file.

If the .env path given doesn't exist, fails.
If the given key doesn't exist in the .env, fails.

### Function: resolve_variables(values, override)

### Function: _walk_to_root(path)

**Description:** Yield directories starting from the given directory up to the root

### Function: find_dotenv(filename, raise_error_if_not_found, usecwd)

**Description:** Search in increasingly higher folders for the given file

Returns path to the file if found, or an empty string otherwise

### Function: load_dotenv(dotenv_path, stream, verbose, override, interpolate, encoding)

**Description:** Parse a .env file and then load all the variables found as environment variables.

Parameters:
    dotenv_path: Absolute or relative path to .env file.
    stream: Text stream (such as `io.StringIO`) with .env content, used if
        `dotenv_path` is `None`.
    verbose: Whether to output a warning the .env file is missing.
    override: Whether to override the system environment variables with the variables
        from the `.env` file.
    encoding: Encoding to be used to read the file.
Returns:
    Bool: True if at least one environment variable is set else False

If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
.env file with it's default parameters. If you need to change the default parameters
of `find_dotenv()`, you can explicitly call `find_dotenv()` and pass the result
to this function as `dotenv_path`.

If the environment variable `PYTHON_DOTENV_DISABLED` is set to a truthy value,
.env loading is disabled.

### Function: dotenv_values(dotenv_path, stream, verbose, interpolate, encoding)

**Description:** Parse a .env file and return its content as a dict.

The returned dict will have `None` values for keys without values in the .env file.
For example, `foo=bar` results in `{"foo": "bar"}` whereas `foo` alone results in
`{"foo": None}`

Parameters:
    dotenv_path: Absolute or relative path to the .env file.
    stream: `StringIO` object with .env content, used if `dotenv_path` is `None`.
    verbose: Whether to output a warning if the .env file is missing.
    encoding: Encoding to be used to read the file.

If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
.env file.

### Function: _is_file_or_fifo(path)

**Description:** Return True if `path` exists and is either a regular file or a FIFO.

### Function: __init__(self, dotenv_path, stream, verbose, encoding, interpolate, override)

### Function: _get_stream(self)

### Function: dict(self)

**Description:** Return dotenv as dict

### Function: parse(self)

### Function: set_as_environment_variables(self)

**Description:** Load the current dotenv as system environment variable.

### Function: get(self, key)

### Function: _is_interactive()

**Description:** Decide whether this is running in a REPL or IPython notebook

### Function: _is_debugger()
