## AI Summary

A file named terminalmanager.py.


## Class: TerminalManager

**Description:** A MultiTerminalManager for use in the notebook webserver

### Function: create(self)

**Description:** Create a new terminal.

### Function: get(self, name)

**Description:** Get terminal 'name'.

### Function: list(self)

**Description:** Get a list of all running terminals.

### Function: get_terminal_model(self, name)

**Description:** Return a JSON-safe dict representing a terminal.
For use in representing terminals in the JSON APIs.

### Function: _check_terminal(self, name)

**Description:** Check a that terminal 'name' exists and raise 404 if not.

### Function: _initialize_culler(self)

**Description:** Start culler if 'cull_inactive_timeout' is greater than zero.
Regardless of that value, set flag that we've been here.

### Function: pre_pty_read_hook(self, ptywclients)

**Description:** The pre-pty read hook.
