## AI Summary

A file named defines.py.


### Function: RaiseIfZero(result, func, arguments)

**Description:** Error checking for most Win32 API calls.

The function is assumed to return an integer, which is C{0} on error.
In that case the C{WindowsError} exception is raised.

### Function: RaiseIfNotZero(result, func, arguments)

**Description:** Error checking for some odd Win32 API calls.

The function is assumed to return an integer, which is zero on success.
If the return value is nonzero the C{WindowsError} exception is raised.

This is mostly useful for free() like functions, where the return value is
the pointer to the memory block on failure or a C{NULL} pointer on success.

### Function: RaiseIfNotErrorSuccess(result, func, arguments)

**Description:** Error checking for Win32 Registry API calls.

The function is assumed to return a Win32 error code. If the code is not
C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.

## Class: GuessStringType

**Description:** Decorator that guesses the correct version (A or W) to call
based on the types of the strings passed as parameters.

Calls the B{ANSI} version if the only string types are ANSI.

Calls the B{Unicode} version if Unicode or mixed string types are passed.

The default if no string arguments are passed depends on the value of the
L{t_default} class variable.

@type fn_ansi: function
@ivar fn_ansi: ANSI version of the API function to call.
@type fn_unicode: function
@ivar fn_unicode: Unicode (wide) version of the API function to call.

@type t_default: type
@cvar t_default: Default string type to use.
    Possible values are:
     - type('') for ANSI
     - type(u'') for Unicode

## Class: DefaultStringType

**Description:** Decorator that uses the default version (A or W) to call
based on the configuration of the L{GuessStringType} decorator.

@see: L{GuessStringType.t_default}

@type fn_ansi: function
@ivar fn_ansi: ANSI version of the API function to call.
@type fn_unicode: function
@ivar fn_unicode: Unicode (wide) version of the API function to call.

### Function: MakeANSIVersion(fn)

**Description:** Decorator that generates an ANSI version of a Unicode (wide) only API call.

@type  fn: callable
@param fn: Unicode (wide) version of the API function to call.

### Function: MakeWideVersion(fn)

**Description:** Decorator that generates a Unicode (wide) version of an ANSI only API call.

@type  fn: callable
@param fn: ANSI version of the API function to call.

## Class: FLOAT128

## Class: M128A

## Class: UNICODE_STRING

## Class: GUID

## Class: LIST_ENTRY

## Class: WinDllHook

## Class: WinFuncHook

## Class: WinCallHook

### Function: __init__(self, fn_ansi, fn_unicode)

**Description:** @type  fn_ansi: function
@param fn_ansi: ANSI version of the API function to call.
@type  fn_unicode: function
@param fn_unicode: Unicode (wide) version of the API function to call.

### Function: __call__(self)

### Function: __init__(self, fn_ansi, fn_unicode)

**Description:** @type  fn_ansi: function
@param fn_ansi: ANSI version of the API function to call.
@type  fn_unicode: function
@param fn_unicode: Unicode (wide) version of the API function to call.

### Function: __call__(self)

### Function: wrapper()

### Function: wrapper()

### Function: __getattr__(self, name)

### Function: __init__(self, name)

### Function: __getattr__(self, name)

### Function: __init__(self, dllname, funcname)

### Function: __copy_attribute(self, attribute)

### Function: __call__(self)
