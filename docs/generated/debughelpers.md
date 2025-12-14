## AI Summary

A file named debughelpers.py.


## Class: UnexpectedUnicodeError

**Description:** Raised in places where we want some better error reporting for
unexpected unicode or binary data.

## Class: DebugFilesKeyError

**Description:** Raised from request.files during debugging.  The idea is that it can
provide a better error message than just a generic KeyError/BadRequest.

## Class: FormDataRoutingRedirect

**Description:** This exception is raised in debug mode if a routing redirect
would cause the browser to drop the method or body. This happens
when method is not GET, HEAD or OPTIONS and the status code is not
307 or 308.

### Function: attach_enctype_error_multidict(request)

**Description:** Patch ``request.files.__getitem__`` to raise a descriptive error
about ``enctype=multipart/form-data``.

:param request: The request to patch.
:meta private:

### Function: _dump_loader_info(loader)

### Function: explain_template_loading_attempts(app, template, attempts)

**Description:** This should help developers understand what failed

### Function: __init__(self, request, key)

### Function: __str__(self)

### Function: __init__(self, request)

## Class: newcls

### Function: __getitem__(self, key)
