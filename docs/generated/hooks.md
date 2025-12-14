## AI Summary

A file named hooks.py.


### Function: editor(self, filename, linenum, wait)

**Description:** Open the default editor at the given filename and linenumber.

This is IPython's default editor hook, you can use it as an example to
write your own modified one.  To set your own editor function as the
new editor hook, call ip.set_hook('editor',yourfunc).

### Function: synchronize_with_editor(self, filename, linenum, column)

## Class: CommandChainDispatcher

**Description:** Dispatch calls to a chain of commands until some func can handle it

Usage: instantiate, execute "add" to add commands (with optional
priority), execute normally via f() calling mechanism.

### Function: show_in_pager(self, data, start, screen_lines)

**Description:** Run a string through pager 

### Function: pre_prompt_hook(self)

**Description:** Run before displaying the next prompt

Use this e.g. to display output from asynchronous operations (in order
to not mess up text entry)

### Function: clipboard_get(self)

**Description:** Get text from the clipboard.
    

### Function: __init__(self, commands)

### Function: __call__(self)

**Description:** Command chain is called just like normal func.

This will call all funcs in chain with the same args as were given to
this function, and return the result of first func that didn't raise
TryNext

### Function: __str__(self)

### Function: add(self, func, priority)

**Description:** Add a func to the cmd chain with given priority 

### Function: __iter__(self)

**Description:** Return all objects in chain.

Handy if the objects are not callable.
