## AI Summary

A file named ipkernel.py.


## Class: InProcessKernel

**Description:** An in-process kernel.

## Class: InProcessInteractiveShell

**Description:** An in-process interactive shell.

### Function: _default_iopub_thread(self)

### Function: _default_iopub_socket(self)

### Function: __init__(self)

**Description:** Initialize the kernel.

### Function: start(self)

**Description:** Override registration of dispatchers for streams.

### Function: _abort_queues(self, subshell_id)

**Description:** The in-process kernel doesn't abort requests.

### Function: _input_request(self, prompt, ident, parent, password)

### Function: _redirected_io(self)

**Description:** Temporarily redirect IO to the kernel.

### Function: _io_dispatch(self, change)

**Description:** Called when a message is sent to the IO socket.

### Function: _default_log(self)

### Function: _default_session(self)

### Function: _default_shell_class(self)

### Function: _default_stdout(self)

### Function: _default_stderr(self)

### Function: enable_gui(self, gui)

**Description:** Enable GUI integration for the kernel.

### Function: enable_matplotlib(self, gui)

**Description:** Enable matplotlib integration for the kernel.

### Function: enable_pylab(self, gui, import_all)

**Description:** Activate pylab support at runtime.
