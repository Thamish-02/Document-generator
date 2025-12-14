## AI Summary

A file named process.py.


### Function: which(command, env)

**Description:** Get the full path to a command.

Parameters
----------
command: str
    The command name or path.
env: dict, optional
    The environment variables, defaults to `os.environ`.

## Class: Process

**Description:** A wrapper for a child process.

## Class: WatchHelper

**Description:** A process helper for a watch process.

### Function: list2cmdline(cmd_list)

**Description:** Shim for list2cmdline on posix.

### Function: __init__(self, cmd, logger, cwd, kill_event, env, quiet)

**Description:** Start a subprocess that can be run asynchronously.

Parameters
----------
cmd: list
    The command to run.
logger: :class:`~logger.Logger`, optional
    The logger instance.
cwd: string, optional
    The cwd of the process.
env: dict, optional
    The environment for the process.
kill_event: :class:`~threading.Event`, optional
    An event used to kill the process operation.
quiet: bool, optional
    Whether to suppress output.

### Function: terminate(self)

**Description:** Terminate the process and return the exit code.

### Function: wait(self)

**Description:** Wait for the process to finish.

Returns
-------
The process exit code.

### Function: wait_async(self)

**Description:** Asynchronously wait for the process to finish.

### Function: _create_process(self)

**Description:** Create the process.

### Function: _cleanup(cls)

**Description:** Clean up the started subprocesses at exit.

### Function: get_log(self)

**Description:** Get our logger.

### Function: __init__(self, cmd, startup_regex, logger, cwd, kill_event, env)

**Description:** Initialize the process helper.

Parameters
----------
cmd: list
    The command to run.
startup_regex: string
    The regex to wait for at startup.
logger: :class:`~logger.Logger`, optional
    The logger instance.
cwd: string, optional
    The cwd of the process.
env: dict, optional
    The environment for the process.
kill_event: callable, optional
    A function to call to check if we should abort.

### Function: terminate(self)

**Description:** Terminate the process.

### Function: _read_incoming(self)

**Description:** Run in a thread to read stdout and print

### Function: _create_process(self)

**Description:** Create the watcher helper process.
