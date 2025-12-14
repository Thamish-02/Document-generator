## AI Summary

A file named _process_posix.py.


## Class: ProcessHandler

**Description:** Execute subprocesses under the control of pexpect.
    

### Function: check_pid(pid)

### Function: sh(self)

### Function: __init__(self, logfile, read_timeout, terminate_timeout)

**Description:** Arguments are used for pexpect calls.

### Function: getoutput(self, cmd)

**Description:** Run a command and return its stdout/stderr as a string.

Parameters
----------
cmd : str
    A command to be executed in the system shell.

Returns
-------
output : str
    A string containing the combination of stdout and stderr from the
subprocess, in whatever order the subprocess originally wrote to its
file descriptors (so the order of the information in this string is the
correct order as would be seen if running the command in a terminal).

### Function: getoutput_pexpect(self, cmd)

**Description:** Run a command and return its stdout/stderr as a string.

Parameters
----------
cmd : str
    A command to be executed in the system shell.

Returns
-------
output : str
    A string containing the combination of stdout and stderr from the
subprocess, in whatever order the subprocess originally wrote to its
file descriptors (so the order of the information in this string is the
correct order as would be seen if running the command in a terminal).

### Function: system(self, cmd)

**Description:** Execute a command in a subshell.

Parameters
----------
cmd : str
    A command to be executed in the system shell.

Returns
-------
int : child's exitstatus
