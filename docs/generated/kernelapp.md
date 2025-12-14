## AI Summary

A file named kernelapp.py.


## Class: KernelApp

**Description:** Launch a kernel by name in a local subprocess.

### Function: initialize(self, argv)

**Description:** Initialize the application.

### Function: setup_signals(self)

**Description:** Shutdown on SIGTERM or SIGINT (Ctrl-C)

### Function: shutdown(self, signo)

**Description:** Shut down the application.

### Function: log_connection_info(self)

**Description:** Log the connection info for the kernel.

### Function: _record_started(self)

**Description:** For tests, create a file to indicate that we've started

Do not rely on this except in our own tests!

### Function: start(self)

**Description:** Start the application.

### Function: shutdown_handler(signo, frame)
