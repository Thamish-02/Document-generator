## AI Summary

A file named stub_value.py.


## Class: StubModuleValue

## Class: StubModuleContext

## Class: TypingModuleWrapper

## Class: TypingModuleContext

## Class: StubFilter

## Class: VersionInfo

### Function: __init__(self, non_stub_value_set)

### Function: is_stub(self)

### Function: sub_modules_dict(self)

**Description:** We have to overwrite this, because it's possible to have stubs that
don't have code for all the child modules. At the time of writing this
there are for example no stubs for `json.tool`.

### Function: _get_stub_filters(self, origin_scope)

### Function: get_filters(self, origin_scope)

### Function: _as_context(self)

### Function: get_filters(self, until_position, origin_scope)

### Function: get_filters(self)

### Function: _as_context(self)

### Function: get_filters(self)

### Function: _is_name_reachable(self, name)
