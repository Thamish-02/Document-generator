## AI Summary

A file named browser_check.py.


## Class: LogErrorHandler

**Description:** A handler that exits with 1 on a logged error.

### Function: run_test(app, func)

**Description:** Synchronous entry point to run a test function.
func is a function that accepts an app url as a parameter and returns a result.
func can be synchronous or asynchronous.  If it is synchronous, it will be run
in a thread, so asynchronous is preferred.

### Function: run_browser_sync(url)

**Description:** Run the browser test and return an exit code.

## Class: BrowserApp

**Description:** An app the launches JupyterLab and waits for it to start up, checking for
JS console errors, JS errors, and Python logged errors.

### Function: _jupyter_server_extension_points()

### Function: _jupyter_server_extension_paths()

### Function: __init__(self)

### Function: filter(self, record)

### Function: emit(self, record)

### Function: initialize_settings(self)

### Function: initialize_handlers(self)

### Function: func()
