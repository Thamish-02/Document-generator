## AI Summary

A file named _process_win32.py.


## Class: AvoidUNCPath

**Description:** A context manager to protect command execution from UNC paths.

In the Win32 API, commands can't be invoked with the cwd being a UNC path.
This context manager temporarily changes directory to the 'C:' drive on
entering, and restores the original working directory on exit.

The context manager returns the starting working directory *if* it made a
change and None otherwise, so that users can apply the necessary adjustment
to their system calls in the event of a change.

Examples
--------
::
    cmd = 'dir'
    with AvoidUNCPath() as path:
        if path is not None:
            cmd = '"pushd %s &&"%s' % (path, cmd)
        os.system(cmd)

### Function: _system_body(p)

**Description:** Callback for _system.

### Function: system(cmd)

**Description:** Win32 version of os.system() that works with network shares.

Note that this implementation returns None, as meant for use in IPython.

Parameters
----------
cmd : str or list
    A command to be executed in the system shell.

Returns
-------
int : child process' exit code.

### Function: getoutput(cmd)

**Description:** Return standard output of executing cmd in a shell.

Accepts the same arguments as os.system().

Parameters
----------
cmd : str or list
    A command to be executed in the system shell.

Returns
-------
stdout : str

### Function: check_pid(pid)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: stdout_read()

### Function: stderr_read()

### Function: arg_split(commandline, posix, strict)

**Description:** Split a command line's arguments in a shell-like manner.

This is a special version for windows that use a ctypes call to CommandLineToArgvW
to do the argv splitting. The posix parameter is ignored.

If strict=False, process_common.arg_split(...strict=False) is used instead.
