## AI Summary

A file named gtkembed.py.


## Class: GTKEmbed

**Description:** A class to embed a kernel into the GTK main event loop.

### Function: __init__(self, kernel)

**Description:** Initialize the embed.

### Function: start(self)

**Description:** Starts the GTK main event loop and sets our kernel startup routine.

### Function: _wire_kernel(self)

**Description:** Initializes the kernel inside GTK.

This is meant to run only once at startup, so it does its job and
returns False to ensure it doesn't get run again by GTK.

### Function: iterate_kernel(self)

**Description:** Run one iteration of the kernel and return True.

GTK timer functions must return True to be called again, so we make the
call to :meth:`do_one_iteration` and then return True for GTK.

### Function: stop(self)

**Description:** Stop the embed.

### Function: _hijack_gtk(self)

**Description:** Hijack a few key functions in GTK for IPython integration.

Modifies pyGTK's main and main_quit with a dummy so user code does not
block IPython.  This allows us to use %run to run arbitrary pygtk
scripts from a long-lived IPython session, and when they attempt to
start or stop

Returns
-------
The original functions that have been hijacked:
- gtk.main
- gtk.main_quit

### Function: dummy()

**Description:** No-op.
