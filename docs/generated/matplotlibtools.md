## AI Summary

A file named matplotlibtools.py.


### Function: do_enable_gui(guiname)

### Function: find_gui_and_backend()

**Description:** Return the gui and mpl backend.

### Function: _get_major_version(module)

### Function: _get_minor_version(module)

### Function: is_interactive_backend(backend)

**Description:** Check if backend is interactive

### Function: patch_use(enable_gui_function)

**Description:** Patch matplotlib function 'use'

### Function: patch_is_interactive()

**Description:** Patch matplotlib function 'use'

### Function: activate_matplotlib(enable_gui_function)

**Description:** Set interactive to True for interactive backends.
enable_gui_function - Function which enables gui, should be run in the main thread.

### Function: flag_calls(func)

**Description:** Wrap a function to detect and flag when it gets called.

This is a decorator which takes a function and wraps it in a function with
a 'called' attribute. wrapper.called is initialized to False.

The wrapper.called attribute is set to False right before each call to the
wrapped function, so if the call fails it remains False.  After the call
completes, wrapper.called is set to True and the output is returned.

Testing for truth in wrapper.called allows you to determine if a call to
func() was attempted and succeeded.

### Function: activate_pylab()

### Function: activate_pyplot()

### Function: patched_use()

### Function: patched_is_interactive()

### Function: wrapper()
