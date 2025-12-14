## AI Summary

A file named terminal.py.


### Function: toggle_set_term_title(val)

**Description:** Control whether set_term_title is active or not.

set_term_title() allows writing to the console titlebar.  In embedded
widgets this can cause problems, so this call can be used to toggle it on
or off as needed.

The default state of the module is for the function to be disabled.

Parameters
----------
val : bool
    If True, set_term_title() actually writes to the terminal (using the
    appropriate platform-specific module).  If False, it is a no-op.

### Function: _set_term_title()

**Description:** Dummy no-op.

### Function: _restore_term_title()

### Function: _set_term_title_xterm(title)

**Description:** Change virtual terminal title in xterm-workalikes 

### Function: _restore_term_title_xterm()

### Function: set_term_title(title)

**Description:** Set terminal title using the necessary platform-dependent calls.

### Function: restore_term_title()

**Description:** Restore, if possible, terminal title to the original state

### Function: freeze_term_title()

### Function: get_terminal_size(defaultx, defaulty)

### Function: _term_clear()

### Function: _term_clear()

### Function: _term_clear()

### Function: _set_term_title(title)

**Description:** Set terminal title using ctypes to access the Win32 APIs.
