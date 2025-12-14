## AI Summary

A file named consoleapp.py.


## Class: JupyterConsoleApp

**Description:** The base Jupyter console application.

## Class: IPythonConsoleApp

**Description:** An app to manage an ipython console.

### Function: _connection_file_default(self)

### Function: build_kernel_argv(self, argv)

**Description:** build argv to be passed to kernel subprocess

Override in subclasses if any args should be passed to the kernel

### Function: init_connection_file(self)

**Description:** find the connection file, and load the info if found.

The current working directory and the current profile's security
directory will be searched for the file if it is not given by
absolute path.

When attempting to connect to an existing kernel and the `--existing`
argument does not match an existing file, it will be interpreted as a
fileglob, and the matching file in the current profile's security dir
with the latest access time will be used.

After this method is called, self.connection_file contains the *full path*
to the connection file, never just its name.

### Function: init_ssh(self)

**Description:** set up ssh tunnels, if needed.

### Function: _new_connection_file(self)

### Function: init_kernel_manager(self)

**Description:** Initialize the kernel manager.

### Function: init_kernel_client(self)

**Description:** Initialize the kernel client.

### Function: initialize(self, argv)

**Description:** Classes which mix this class in should call:
       JupyterConsoleApp.initialize(self,argv)

### Function: __init__(self)

**Description:** Initialize the app.
