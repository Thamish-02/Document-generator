## AI Summary

A file named types.py.


## Class: SessionStatus

**Description:** States in which a language server session can be

## Class: MessageScope

**Description:** Scopes for message listeners

## Class: MessageListener

**Description:** A base listener implementation

## Class: HasListeners

## Class: LanguageServerManagerAPI

**Description:** Public API that can be used for python-based spec finders and listeners

## Class: SpecBase

**Description:** Base for a spec finder that returns a spec for starting a language server

## Class: HandlerListenerCallback

### Function: __init__(self, listener, language_server, method)

### Function: wants(self, message, language_server)

**Description:** whether this listener wants a particular message

`method` is currently the only message content discriminator, but not
all messages will have a `method`

### Function: __repr__(self)

### Function: register_message_listener(cls, scope, language_server, method)

**Description:** register a listener for language server protocol messages

### Function: unregister_message_listener(cls, listener)

**Description:** unregister a listener for language server protocol messages

### Function: find_node_module(self)

**Description:** look through the node_module roots to find the given node module

### Function: _default_nodejs(self)

### Function: _npm_prefix(self, npm)

### Function: _default_node_roots(self)

**Description:** get the "usual suspects" for where `node_modules` may be found

- where this was launch (usually the same as NotebookApp.notebook_dir)
- the JupyterLab staging folder (if available)
- wherever conda puts it
- wherever some other conventions put it

### Function: is_installed(self, mgr)

**Description:** Whether the language server is installed or not.

This method may become abstract in the next major release.

### Function: __call__(self, mgr)

### Function: __call__(self, scope, message, language_server, manager)

### Function: inner(listener)
