## AI Summary

A file named session.py.


## Class: LanguageServerSession

**Description:** Manage a session for a connection to a language server

### Function: __init__(self)

**Description:** set up the required traitlets and exit behavior for a session

### Function: __repr__(self)

### Function: to_json(self)

### Function: initialize(self)

**Description:** (re)initialize a language server session

### Function: stop(self)

**Description:** clean up all of the state of the session

### Function: _on_handlers(self, change)

**Description:** re-initialize if someone starts listening, or stop if nobody is

### Function: write(self, message)

**Description:** wrapper around the write queue to keep it mostly internal

### Function: now(self)

### Function: init_process(self)

**Description:** start the language server subprocess

### Function: init_queues(self)

**Description:** create the queues

### Function: init_reader(self)

**Description:** create the stdout reader (from the language server)

### Function: init_writer(self)

**Description:** create the stdin writer (to the language server)

### Function: substitute_env(self, env, base)
