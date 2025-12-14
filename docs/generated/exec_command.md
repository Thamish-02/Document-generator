## AI Summary

A file named exec_command.py.


### Function: filepath_from_subprocess_output(output)

**Description:** Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

Inherited from `exec_command`, and possibly incorrect.

### Function: forward_bytes_to_stdout(val)

**Description:** Forward bytes from a subprocess call to the console, without attempting to
decode them.

The assumption is that the subprocess call already returned bytes in
a suitable encoding.

### Function: temp_file_name()

### Function: get_pythonexe()

### Function: find_executable(exe, path, _cache)

**Description:** Return full path of a executable or None.

Symbolic links are not followed.

### Function: _preserve_environment(names)

### Function: _update_environment()

### Function: exec_command(command, execute_in, use_shell, use_tee, _with_python)

**Description:** Return (status,output) of executed command.

.. deprecated:: 1.17
    Use subprocess.Popen instead

Parameters
----------
command : str
    A concatenated string of executable and arguments.
execute_in : str
    Before running command ``cd execute_in`` and after ``cd -``.
use_shell : {bool, None}, optional
    If True, execute ``sh -c command``. Default None (True)
use_tee : {bool, None}, optional
    If True use tee. Default None (True)


Returns
-------
res : str
    Both stdout and stderr messages.

Notes
-----
On NT, DOS systems the returned status is correct for external commands.
Wild cards will not work for non-posix systems or when use_shell=0.

### Function: _exec_command(command, use_shell, use_tee)

**Description:** Internal workhorse for exec_command().

### Function: _quote_arg(arg)

**Description:** Quote the argument for safe use in a shell command line.
