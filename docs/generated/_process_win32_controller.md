## AI Summary

A file named _process_win32_controller.py.


## Class: SECURITY_ATTRIBUTES

## Class: STARTUPINFO

## Class: PROCESS_INFORMATION

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

## Class: Win32ShellCommandController

**Description:** Runs a shell command in a 'with' context.

This implementation is Win32-specific.

Example:
    # Runs the command interactively with default console stdin/stdout
    with ShellCommandController('python -i') as scc:
        scc.run()

    # Runs the command using the provided functions for stdin/stdout
    def my_stdout_func(s):
        # print or save the string 's'
        write_to_stdout(s)
    def my_stdin_func():
        # If input is available, return it as a string.
        if input_available():
            return get_input()
        # If no input available, return None after a short delay to
        # keep from blocking.
        else:
            time.sleep(0.01)
            return None
  
    with ShellCommandController('python -i') as scc:
        scc.run(my_stdout_func, my_stdin_func)

### Function: system(cmd)

**Description:** Win32 version of os.system() that works with network shares.

Note that this implementation returns None, as meant for use in IPython.

Parameters
----------
cmd : str
    A command to be executed in the system shell.

Returns
-------
None : we explicitly do NOT return the subprocess status code, as this
utility is meant to be used extensively in IPython, where any return value
would trigger : func:`sys.displayhook` calls.

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, cmd, mergeout)

**Description:** Initializes the shell command controller.

The cmd is the program to execute, and mergeout is
whether to blend stdout and stderr into one output
in stdout. Merging them together in this fashion more
reliably keeps stdout and stderr in the correct order
especially for interactive shell usage.

### Function: __enter__(self)

### Function: _stdin_thread(self, handle, hprocess, func, stdout_func)

### Function: _stdout_thread(self, handle, func)

### Function: run(self, stdout_func, stdin_func, stderr_func)

**Description:** Runs the process, using the provided functions for I/O.

The function stdin_func should return strings whenever a
character or characters become available.
The functions stdout_func and stderr_func are called whenever
something is printed to stdout or stderr, respectively.
These functions are called from different threads (but not
concurrently, because of the GIL).

### Function: _stdin_raw_nonblock(self)

**Description:** Use the raw Win32 handle of sys.stdin to do non-blocking reads

### Function: _stdin_raw_block(self)

**Description:** Use a blocking stdin read

### Function: _stdout_raw(self, s)

**Description:** Writes the string to stdout

### Function: _stderr_raw(self, s)

**Description:** Writes the string to stdout

### Function: _run_stdio(self)

**Description:** Runs the process using the system standard I/O.

IMPORTANT: stdin needs to be asynchronous, so the Python
           sys.stdin object is not used. Instead,
           msvcrt.kbhit/getwch are used asynchronously.

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: create_pipe(uninherit)

**Description:** Creates a Windows pipe, which consists of two handles.

The 'uninherit' parameter controls which handle is not
inherited by the child process.
