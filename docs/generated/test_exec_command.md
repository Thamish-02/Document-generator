## AI Summary

A file named test_exec_command.py.


## Class: redirect_stdout

**Description:** Context manager to redirect stdout for exec_command test.

## Class: redirect_stderr

**Description:** Context manager to redirect stderr for exec_command test.

## Class: emulate_nonposix

**Description:** Context manager to emulate os.name != 'posix' 

### Function: test_exec_command_stdout()

### Function: test_exec_command_stderr()

## Class: TestExecCommand

### Function: __init__(self, stdout)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, stderr)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: __init__(self, osname)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, traceback)

### Function: setup_method(self)

### Function: check_nt(self)

### Function: check_posix(self)

### Function: check_basic(self)

### Function: check_execute_in(self)

### Function: test_basic(self)
