## AI Summary

A file named compilerop.py.


### Function: code_name(code, number)

**Description:** Compute a (probably) unique name for code for caching.

This now expects code to be unicode.

## Class: CachingCompiler

**Description:** A compiler that caches code compiled from interactive statements.
    

### Function: check_linecache_ipython()

**Description:** Deprecated since IPython 8.6.  Call linecache.checkcache() directly.

It was already not necessary to call this function directly.  If no
CachingCompiler had been created, this function would fail badly.  If
an instance had been created, this function would've been monkeypatched
into place.

As of IPython 8.6, the monkeypatching has gone away entirely.  But there
were still internal callers of this function, so maybe external callers
also existed?

### Function: __init__(self)

### Function: ast_parse(self, source, filename, symbol)

**Description:** Parse code to an AST with the current compiler flags active.

Arguments are exactly the same as ast.parse (in the standard library),
and are passed to the built-in compile function.

### Function: reset_compiler_flags(self)

**Description:** Reset compiler flags to default state.

### Function: compiler_flags(self)

**Description:** Flags currently active in the compilation process.
        

### Function: get_code_name(self, raw_code, transformed_code, number)

**Description:** Compute filename given the code, and the cell number.

Parameters
----------
raw_code : str
    The raw cell code.
transformed_code : str
    The executable Python source code to cache and compile.
number : int
    A number which forms part of the code's name. Used for the execution
    counter.

Returns
-------
The computed filename.

### Function: format_code_name(self, name)

**Description:** Return a user-friendly label and name for a code block.

Parameters
----------
name : str
    The name for the code block returned from get_code_name

Returns
-------
A (label, name) pair that can be used in tracebacks, or None if the default formatting should be used.

### Function: cache(self, transformed_code, number, raw_code)

**Description:** Make a name for a block of code, and cache the code.

Parameters
----------
transformed_code : str
    The executable Python source code to cache and compile.
number : int
    A number which forms part of the code's name. Used for the execution
    counter.
raw_code : str
    The raw code before transformation, if None, set to `transformed_code`.

Returns
-------
The name of the cached code (as a string). Pass this as the filename
argument to compilation, so that tracebacks are correctly hooked up.

### Function: extra_flags(self, flags)
